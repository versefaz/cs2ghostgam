# core/ml/feature_engineer.py
import yaml
import os


def load_feature_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        # default minimal config
        return {"features": {"baseline": ["team_elo", "map_winrate", "form_5"]}}


class FeatureEngineer:
    def __init__(self, config: dict = None):
        self.config = config or load_feature_config(os.getenv("FEATURE_CONFIG", "configs/features/cs2.yaml"))
        # validate keys
        if "features" not in self.config:
            self.config["features"] = {"baseline": ["team_elo", "map_winrate", "form_5"]}

    def extract_features(self, match_data: dict) -> list:
        """Extract features from match data"""
        features = []
        
        # Basic team features
        features.extend([
            match_data.get('team1_ranking', 50),
            match_data.get('team2_ranking', 50),
            match_data.get('team1_form', 0.5),
            match_data.get('team2_form', 0.5),
        ])
        
        # Odds features
        features.extend([
            match_data.get('team1_odds', 2.0),
            match_data.get('team2_odds', 2.0),
            match_data.get('bookmaker_count', 1),
        ])
        
        # H2H features
        h2h_total = match_data.get('h2h_total', 1)
        features.extend([
            match_data.get('h2h_team1_wins', 0) / max(h2h_total, 1),
            match_data.get('h2h_team2_wins', 0) / max(h2h_total, 1),
        ])
        
        return features
