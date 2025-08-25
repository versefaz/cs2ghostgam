from typing import Dict, Optional
from datetime import datetime

from config import settings
from utils.kelly_criterion import KellyCriterion


class SignalGenerator:
    def __init__(self):
        self.kelly = KellyCriterion()

    def generate(self, match: Dict, prediction: Dict) -> Optional[Dict]:
        conf = float(prediction.get('confidence', 0))
        if conf < settings.MIN_CONFIDENCE:
            return None
        odds = float(prediction.get('odds', 2.0))
        kelly_fraction = self.kelly.calculate(conf, odds, 1.0)
        bet_fraction = min(kelly_fraction * 0.25, settings.MAX_BET_SIZE)

        strength = 'weak'
        if conf >= 0.85:
            strength = 'strong'
        elif conf >= 0.75:
            strength = 'medium'

        return {
            'timestamp': datetime.now(),
            'match': f"{match.get('team1','?')} vs {match.get('team2','?')}",
            'prediction': prediction.get('predicted_winner'),
            'market': 'match_winner',
            'odds': odds,
            'confidence': conf,
            'bet_size': bet_fraction,
            'strength': strength,
            'reasons': [],
        }
