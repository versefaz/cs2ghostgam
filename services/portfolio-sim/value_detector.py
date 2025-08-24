from typing import Dict, List

class ValueBetDetector:
    """ตรวจหา Value Bets แบบ realtime"""
    def calculate_confidence(self, edge: float, model_prob: float) -> float:
        return max(0.0, min(1.0, 0.5 * edge + 0.5 * model_prob))

    def find_value_bets(self, predictions: Dict, odds: Dict) -> List[Dict]:
        value_bets: List[Dict] = []
        for market in ['moneyline', 'handicap', 'totals']:
            if market not in odds or market not in predictions:
                continue
            for selection, book_odds in odds[market].items():
                model_prob = predictions[market].get(selection, 0.0)
                if book_odds <= 1:
                    continue
                implied_prob = 1.0 / book_odds
                edge = model_prob - implied_prob
                ev = (model_prob * (book_odds - 1)) - (1 - model_prob)
                if ev > 0.05:
                    value_bets.append({
                        'market': market,
                        'selection': selection,
                        'model_prob': model_prob,
                        'odds': book_odds,
                        'edge': edge,
                        'ev': ev,
                        'confidence': self.calculate_confidence(edge, model_prob),
                    })
        return sorted(value_bets, key=lambda x: x['ev'], reverse=True)
