import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

# aiohttp import with fallback handling
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import redis
except ImportError:
    redis = None


class BettingSystemIntegration:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.prediction_api = "http://localhost:8000/predict"
        self.scraper_api = "http://localhost:8001/matches"

    async def fetch_live_matches(self) -> List[Dict]:
        if aiohttp is None:
            return []
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.scraper_api) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('matches', [])
            except Exception as e:
                print(f"Error fetching matches: {e}")
            return []

    async def get_predictions(self, match_data: Dict) -> Dict:
        if aiohttp is None:
            return {}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.prediction_api, json=match_data) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception as e:
                print(f"Error getting predictions: {e}")
            return {}

    def calculate_kelly(self, prob: float, odds: float) -> float:
        q = 1 - prob
        b = odds - 1
        if b <= 0:
            return 0
        kelly = (prob * b - q) / b
        return max(0.0, kelly)

    def get_signal_strength(self, confidence: float) -> str:
        if confidence >= 0.85:
            return 'strong'
        elif confidence >= 0.75:
            return 'medium'
        else:
            return 'weak'

    def generate_signal(self, match: Dict, prediction: Dict) -> Optional[Dict]:
        confidence = float(prediction.get('confidence', 0))
        if confidence < 0.7:
            return None
        odds = float(prediction.get('odds', 2.0))
        kelly_fraction = self.calculate_kelly(confidence, odds)
        bet_fraction = min(kelly_fraction * 0.25, 0.05)
        signal = {
            'timestamp': datetime.now(),
            'match': f"{match.get('team1','?')} vs {match.get('team2','?')}",
            'prediction': prediction.get('winner'),
            'market': prediction.get('market', 'match_winner'),
            'odds': odds,
            'confidence': confidence,
            'bet_size': bet_fraction,
            'strength': self.get_signal_strength(confidence),
            'reasons': prediction.get('reasons', []),
        }
        return signal

    async def monitor_and_signal(self):
        while True:
            try:
                matches = await self.fetch_live_matches()
                for match in matches:
                    match_id = f"{match.get('team1','')}_{match.get('team2','')}_{match.get('date','')}"
                    if not self.redis_client.exists(f"processed:{match_id}"):
                        prediction = await self.get_predictions(match)
                        if prediction:
                            signal = self.generate_signal(match, prediction)
                            if signal:
                                self.redis_client.lpush('betting_signals', json.dumps(signal, default=str))
                                self.redis_client.setex(f"processed:{match_id}", 86400, "1")
                                print(f"New signal generated: {signal['match']}")
                await asyncio.sleep(60)
            except Exception as e:
                print(f"Error in monitoring: {e}")
                await asyncio.sleep(30)


if __name__ == "__main__":
    system = BettingSystemIntegration()
    asyncio.run(system.monitor_and_signal())
