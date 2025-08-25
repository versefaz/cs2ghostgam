import math
from typing import Dict, List
import numpy as np
import pandas as pd

from config import settings


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class PredictionModel:
    """
    Lightweight prediction wrapper. If a trained model is not present, it uses a
    heuristic combining rating/form/odds to produce calibrated probabilities.
    """

    def __init__(self):
        self.feature_columns = settings.FEATURE_COLUMNS

    def extract_features(self, match: Dict) -> Dict:
        features: Dict = {}
        t1_stats = match.get('team1_stats', {})
        t2_stats = match.get('team2_stats', {})

        # Ratings: lower rank number => better; convert to score
        t1_rank = float(t1_stats.get('world_ranking', 50) or 50)
        t2_rank = float(t2_stats.get('world_ranking', 50) or 50)
        t1_rating = max(0.0, 100 - t1_rank)
        t2_rating = max(0.0, 100 - t2_rank)
        features['team1_rating'] = t1_rating
        features['team2_rating'] = t2_rating

        # Recent form
        t1_form_list: List[str] = t1_stats.get('recent_form', []) or []
        t2_form_list: List[str] = t2_stats.get('recent_form', []) or []
        features['team1_form'] = (t1_form_list.count('W') / max(len(t1_form_list), 1)) if t1_form_list else 0.5
        features['team2_form'] = (t2_form_list.count('W') / max(len(t2_form_list), 1)) if t2_form_list else 0.5

        # H2H/map placeholders
        features['h2h_score'] = 0.5
        features['map_winrate_diff'] = 0.0

        # Odds features
        odds_t1 = float(match.get('odds_team1') or 2.0)
        odds_t2 = float(match.get('odds_team2') or 2.0)
        features['avg_odds'] = (odds_t1 + odds_t2) / 2.0
        features['odds_movement'] = 0.0
        return features

    def _heuristic_prob(self, features: Dict) -> float:
        # Combine rating and form deltas
        rating_delta = features['team1_rating'] - features['team2_rating']
        form_delta = features['team1_form'] - features['team2_form']
        z = 0.02 * rating_delta + 1.5 * form_delta
        return float(sigmoid(z))

    def predict(self, match: Dict) -> Dict:
        features = self.extract_features(match)
        # Probability team1 wins
        p_t1 = self._heuristic_prob(features)
        p_t2 = 1.0 - p_t1

        # Pick side with higher edge vs market odds
        odds_t1 = float(match.get('odds_team1') or 2.0)
        odds_t2 = float(match.get('odds_team2') or 2.0)
        ev_t1 = p_t1 * odds_t1
        ev_t2 = p_t2 * odds_t2

        if ev_t1 >= ev_t2:
            winner = match.get('team1', 'Team1')
            win_prob = p_t1
            odds = odds_t1
            value = ev_t1
        else:
            winner = match.get('team2', 'Team2')
            win_prob = p_t2
            odds = odds_t2
            value = ev_t2

        return {
            'match_id': match.get('match_id'),
            'team': winner,
            'confidence': float(win_prob),
            'win_prob': float(win_prob),
            'odds': float(odds),
            'value': float(value),
            'ev': float(value),
            'predicted_winner': winner,
            'form_diff': float(features['team1_form'] - features['team2_form']),
            'h2h_score': float(features['h2h_score']),
        }

    def batch_predict(self, matches: List[Dict]):
        preds = []
        for m in matches:
            try:
                preds.append(self.predict(m))
            except Exception as e:
                print(f"Prediction error for {m.get('match_id')}: {e}")
        return preds
