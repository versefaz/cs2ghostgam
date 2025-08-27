# core/ml/model_io.py
import pickle
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Any:
    """Load ML model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.debug(f"Could not load model from {model_path}: {e}")
        return None


def save_model(model: Any, model_path: str) -> bool:
    """Save ML model to pickle file"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Successfully saved model to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        return False


class SimpleCS2Model:
    """Simple CS2 prediction model for fallback"""
    
    def __init__(self):
        self.model_version = "1.0"
        self.features = ["team_elo", "map_winrate", "form_5", "h2h_record"]
        
    def predict_match_outcome(self, match_data: dict) -> dict:
        """Simple prediction based on basic features"""
        team1_name = match_data.get('team1_name', 'Team1')
        team2_name = match_data.get('team2_name', 'Team2')
        features = match_data.get('features', [])
        
        # Simple prediction logic
        if len(features) >= 4:
            team1_score = features[0] * 0.3 + features[2] * 0.7  # ranking + form
            team2_score = features[1] * 0.3 + features[3] * 0.7
            
            if team1_score > team2_score:
                predicted_winner = team1_name
                confidence = min(0.65 + abs(team1_score - team2_score) * 0.1, 0.95)
                prob1, prob2 = 0.6, 0.4
            else:
                predicted_winner = team2_name
                confidence = min(0.65 + abs(team2_score - team1_score) * 0.1, 0.95)
                prob1, prob2 = 0.4, 0.6
        else:
            # Random prediction with low confidence
            import random
            predicted_winner = random.choice([team1_name, team2_name])
            confidence = 0.55
            prob1, prob2 = 0.5, 0.5
            
        return {
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'probabilities': {
                team1_name: prob1,
                team2_name: prob2
            },
            'model_version': self.model_version
        }


def create_default_model() -> SimpleCS2Model:
    """Create default CS2 model"""
    return SimpleCS2Model()
