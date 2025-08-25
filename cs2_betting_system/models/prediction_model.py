import os
import math
import pickle
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
        self.model = None
        self._load_model_if_available()

    def _load_model_if_available(self):
        """Try to load a trained model from disk; stay silent on failure."""
        try:
            path = settings.MODEL_PATH
            if path and os.path.exists(path):
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)
        except Exception as e:
            # Keep heuristic fallback if loading fails
            print(f"Model load failed, using heuristic: {e}")

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
        p_t1 = None
        if self.model is not None:
            try:
                # Build feature vector in configured order; missing features default to 0.0
                row = [[float(features.get(col, 0.0)) for col in self.feature_columns]]
                X = pd.DataFrame(row, columns=self.feature_columns)
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X)
                    # Use probability of positive class; assume second column
                    p_t1 = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
                elif hasattr(self.model, 'decision_function'):
                    z = float(self.model.decision_function(X)[0])
                    p_t1 = float(sigmoid(z))
                elif hasattr(self.model, 'predict'):
                    y = float(self.model.predict(X)[0])
                    # If only class label, approximate as high/low confidence
                    p_t1 = 0.7 if y >= 0.5 else 0.3
            except Exception as e:
                print(f"Model inference failed, using heuristic: {e}")
                p_t1 = None

        if p_t1 is None:
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
