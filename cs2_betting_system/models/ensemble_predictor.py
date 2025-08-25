from typing import Dict, List
import numpy as np
from .prediction_model import PredictionModel

class EnsemblePredictor:
    """Simple ensemble wrapper. Can be extended to load multiple models."""
    def __init__(self, models: List[PredictionModel] = None):
        self.models = models or [PredictionModel()]

    def predict(self, match: Dict) -> Dict:
        preds = [m.predict(match) for m in self.models]
        if not preds:
            return {}
        # Average probabilities and pick consensus winner
        probs = [p['win_prob'] for p in preds]
        avg_prob = float(np.mean(probs))
        # Use first prediction for side selection heuristic
        base = preds[0]
        out = dict(base)
        out['win_prob'] = avg_prob
        out['confidence'] = avg_prob
        return out
