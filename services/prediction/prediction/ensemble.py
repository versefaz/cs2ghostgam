from typing import Any
# Placeholder imports; install heavy deps when training environment is ready
# import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier
from .models.live_predictor import CS2LivePredictor

class BettingEnsemble:
    """Ensemble scaffold (weights set via validation later)."""
    def __init__(self) -> None:
        self.models = {
            # 'lgb': lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01, max_depth=8, objective='binary'),
            # 'xgb': xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=8, tree_method='hist'),
            # 'catboost': CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=8),
            'neural': CS2LivePredictor(),
            # 'tabnet': TabNetClassifier()
        }
        self.weights = [1.0]  # adjust when other models enabled
