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

        # === HLTV World Rankings ===
        t1_rank = float(t1_stats.get('world_ranking', 50) or 50)
        t2_rank = float(t2_stats.get('world_ranking', 50) or 50)
        t1_rating = max(0.0, 100 - t1_rank)
        t2_rating = max(0.0, 100 - t2_rank)
        features['team1_rating'] = t1_rating
        features['team2_rating'] = t2_rating
        features['rating_diff'] = t1_rating - t2_rating

        # === Recent Form (from HLTV processed data) ===
        t1_processed = t1_stats.get('processed', {})
        t2_processed = t2_stats.get('processed', {})
        
        features['team1_form'] = t1_processed.get('recent_win_rate', 0.5)
        features['team2_form'] = t2_processed.get('recent_win_rate', 0.5)
        features['form_diff'] = features['team1_form'] - features['team2_form']
        
        # Momentum scores
        features['team1_momentum'] = t1_processed.get('momentum', 0.0)
        features['team2_momentum'] = t2_processed.get('momentum', 0.0)
        features['momentum_diff'] = features['team1_momentum'] - features['team2_momentum']

        # === Head-to-Head Analysis ===
        h2h_data = match.get('h2h_stats', {})
        if h2h_data and h2h_data.get('total_matches', 0) > 0:
            team1_wins = h2h_data.get('team1_wins', 0)
            total_matches = h2h_data.get('total_matches', 1)
            features['h2h_score'] = team1_wins / total_matches
            features['h2h_matches_played'] = min(total_matches / 20.0, 1.0)  # Normalize to 20 matches
        else:
            features['h2h_score'] = 0.5
            features['h2h_matches_played'] = 0.0

        # === Map-specific Features ===
        map_name = match.get('map', 'TBA')
        if map_name != 'TBA':
            t1_map_stats = t1_stats.get('map_stats', {}).get(map_name, {})
            t2_map_stats = t2_stats.get('map_stats', {}).get(map_name, {})
            
            t1_map_wr = t1_map_stats.get('win_rate', 0.5)
            t2_map_wr = t2_map_stats.get('win_rate', 0.5)
            
            features['team1_map_winrate'] = t1_map_wr
            features['team2_map_winrate'] = t2_map_wr
            features['map_winrate_diff'] = t1_map_wr - t2_map_wr
            
            # Map experience
            t1_map_matches = t1_map_stats.get('matches', 0)
            t2_map_matches = t2_map_stats.get('matches', 0)
            features['team1_map_experience'] = min(t1_map_matches / 50.0, 1.0)
            features['team2_map_experience'] = min(t2_map_matches / 50.0, 1.0)
        else:
            features['team1_map_winrate'] = 0.5
            features['team2_map_winrate'] = 0.5
            features['map_winrate_diff'] = 0.0
            features['team1_map_experience'] = 0.0
            features['team2_map_experience'] = 0.0

        # === Player Performance ===
        t1_players = t1_stats.get('player_stats', [])
        t2_players = t2_stats.get('player_stats', [])
        
        if t1_players:
            features['team1_avg_rating'] = np.mean([p.get('rating', 1.0) for p in t1_players])
            features['team1_avg_kd'] = np.mean([p.get('kd_ratio', 1.0) for p in t1_players])
        else:
            features['team1_avg_rating'] = 1.0
            features['team1_avg_kd'] = 1.0
            
        if t2_players:
            features['team2_avg_rating'] = np.mean([p.get('rating', 1.0) for p in t2_players])
            features['team2_avg_kd'] = np.mean([p.get('kd_ratio', 1.0) for p in t2_players])
        else:
            features['team2_avg_rating'] = 1.0
            features['team2_avg_kd'] = 1.0
            
        features['rating_player_diff'] = features['team1_avg_rating'] - features['team2_avg_rating']
        features['kd_diff'] = features['team1_avg_kd'] - features['team2_avg_kd']

        # === Tournament Context ===
        features['tournament_tier'] = match.get('tournament_tier', 2)
        features['is_lan'] = 1.0 if match.get('is_lan', False) else 0.0
        features['prize_pool'] = min(match.get('prize_pool', 0) / 1000000.0, 1.0)  # Normalize to 1M

        # === Odds Features ===
        odds_t1 = float(match.get('odds_team1') or 2.0)
        odds_t2 = float(match.get('odds_team2') or 2.0)
        features['odds_team1'] = odds_t1
        features['odds_team2'] = odds_t2
        features['avg_odds'] = (odds_t1 + odds_t2) / 2.0
        features['odds_implied_prob_t1'] = 1.0 / odds_t1
        features['odds_implied_prob_t2'] = 1.0 / odds_t2
        
        # Odds movement (if available)
        opening_odds = match.get('opening_odds', {})
        if opening_odds:
            opening_t1 = opening_odds.get('team1', odds_t1)
            opening_t2 = opening_odds.get('team2', odds_t2)
            features['odds_movement_t1'] = odds_t1 - opening_t1
            features['odds_movement_t2'] = odds_t2 - opening_t2
        else:
            features['odds_movement_t1'] = 0.0
            features['odds_movement_t2'] = 0.0
            
        features['odds_movement'] = abs(features['odds_movement_t1']) + abs(features['odds_movement_t2'])
        
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
