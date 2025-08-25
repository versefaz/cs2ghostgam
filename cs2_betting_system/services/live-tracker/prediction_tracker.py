import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List
import redis

from cs2_betting_system.scrapers.live_match_scraper import LiveMatchScraper
from cs2_betting_system.models.ensemble_predictor import EnsemblePredictor
from cs2_betting_system.config import settings

logger = logging.getLogger(__name__)

class EnhancedPredictionTracker:
    def __init__(self):
        self.scraper = LiveMatchScraper()
        self.model = EnsemblePredictor()
        self.redis = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

    async def process_matches_once(self):
        matches = await asyncio.to_thread(self.scraper.scrape_all_sources)
        for m in matches:
            pred = self.model.predict(m)
            record = {
                'match_id': m.get('match_id'),
                'teams': [m.get('team1'), m.get('team2')],
                'status': m.get('status'),
                'timestamp': datetime.now().isoformat(),
                'odds': {'team1': m.get('odds_team1'), 'team2': m.get('odds_team2'), 'source': m.get('odds_source')},
                'prediction': pred,
            }
            key = f"prediction:{record['match_id']}"
            self.redis.setex(key, 3600, json.dumps(record, default=str))
            self.redis.lpush('predictions:recent', json.dumps(record, default=str))
            self.redis.ltrim('predictions:recent', 0, 499)

    async def run(self, interval: int = 60):
        while True:
            try:
                await self.process_matches_once()
            except Exception as e:
                logger.error(f"Tracker loop error: {e}")
            await asyncio.sleep(interval)

async def main():
    logging.basicConfig(level=logging.INFO)
    tracker = EnhancedPredictionTracker()
    await tracker.run(settings.SCRAPE_INTERVAL)

if __name__ == '__main__':
    asyncio.run(main())
