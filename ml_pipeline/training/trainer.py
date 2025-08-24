import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Orchestrate model training, validation, and deployment
    """

    def __init__(self, config: Dict):
        self.config = config
        self.models: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[Any] = None
        self.metrics_history: List[Dict[str, Any]] = []

        # Initialize MLflow
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'mlruns'))
        mlflow.set_experiment(config.get('experiment_name', 'cs2_prediction'))

    def train_model(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: pd.DataFrame,
                    y_val: pd.Series,
                    model_type: str = 'baseline') -> Dict:
        """Train a single model"""
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Get model
            if model_type == 'baseline':
                model = self._get_baseline_model()
            elif model_type == 'advanced':
                model = self._get_advanced_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Train model
            logger.info(f"Training {model_type} model...")
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)

            # Log to MLflow
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Save model
            mlflow.sklearn.log_model(model, model_type)

            # Store model
            self.models[model_type] = {
                'model': model,
                'metrics': metrics,
                'training_date': datetime.now(),
            }

            logger.info(
                f"Model {model_type} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}"
            )
            return metrics

    def _get_baseline_model(self):
        """Get baseline model (RandomForest)"""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

    def _get_advanced_model(self):
        """Get advanced model (XGBoost)"""
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
        )

    def train_ensemble(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: pd.DataFrame,
                       y_val: pd.Series) -> Dict:
        """Train ensemble of models"""
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        import xgboost as xgb
        try:
            import lightgbm as lgb
            lgbm_available = True
        except Exception:
            lgbm_available = False

        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
        ]
        if lgbm_available:
            base_models.append(('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42)))

        ensemble = VotingClassifier(estimators=base_models, voting='soft')
        logger.info("Training ensemble model...")
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_val)
        y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        self.models['ensemble'] = {
            'model': ensemble,
            'metrics': metrics,
            'training_date': datetime.now(),
        }
        logger.info(
            f"Ensemble - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}"
        )
        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = self._get_baseline_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
            for k, v in metrics.items():
                cv_scores[k].append(v)
            logger.info(
                f"Fold {fold} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}"
            )
        cv_results = {f"{metric}_mean": float(np.mean(scores)) for metric, scores in cv_scores.items()}
        cv_results.update({f"{metric}_std": float(np.std(scores)) for metric, scores in cv_scores.items()})
        logger.info(
            f"CV Mean Accuracy: {cv_results['accuracy_mean']:.4f} (+/- {cv_results['accuracy_std']:.4f})"
        )
        return cv_results

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) == 2 else 0,
        }
        metrics.update(self._calculate_betting_metrics(y_true, y_pred_proba))
        return metrics

    def _calculate_betting_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        """Betting-oriented metrics (placeholder)"""
        kelly_fraction = self._calculate_kelly_fraction(y_pred_proba)
        ev = self._calculate_expected_value(y_true, y_pred_proba)
        calibration_error = self._calculate_calibration_error(y_true, y_pred_proba)
        return {
            'kelly_fraction': float(np.mean(kelly_fraction)),
            'expected_value': float(ev),
            'calibration_error': float(calibration_error),
            'roi': float(self._calculate_roi(y_true, y_pred_proba)),
        }

    def _calculate_kelly_fraction(self, probabilities: np.ndarray, odds: Optional[np.ndarray] = None) -> np.ndarray:
        if odds is None:
            odds = 1 / np.clip(probabilities, 1e-6, 1 - 1e-6)
        b = odds - 1
        p = probabilities
        q = 1 - p
        kelly = (p * b - q) / np.clip(b, 1e-6, None)
        return np.clip(kelly, 0, 0.25)

    def _calculate_expected_value(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        stake = 1.0
        returns = np.where(y_true == 1, stake * (1 / np.clip(y_pred_proba, 1e-6, 1 - 1e-6) - 1), -stake)
        return float(np.mean(returns))

    def _calculate_calibration_error(self, y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        calibration_error = 0.0
        for i in range(n_bins):
            bin_mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = float(np.mean(y_true[bin_mask]))
                bin_confidence = float(np.mean(y_pred_proba[bin_mask]))
                bin_weight = float(np.sum(bin_mask) / len(y_pred_proba))
                calibration_error += bin_weight * abs(bin_accuracy - bin_confidence)
        return float(calibration_error)

    def _calculate_roi(self, y_true: pd.Series, y_pred_proba: np.ndarray, threshold: float = 0.55) -> float:
        bet_mask = y_pred_proba > threshold
        if np.sum(bet_mask) == 0:
            return 0.0
        returns = np.where(y_true[bet_mask] == 1, 1.0, -1.0)
        return float((np.sum(returns) / np.sum(bet_mask)) * 100)

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series,
                              model_type: str = 'xgboost') -> Dict:
        import optuna
        from optuna.integration import MLflowCallback

        def objective(trial):
            if model_type == 'xgboost':
                import xgboost as xgb
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    'random_state': 42,
                }
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            elif model_type == 'lightgbm':
                import lightgbm as lgb
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'random_state': 42,
                }
                model = lgb.LGBMClassifier(**params, verbosity=-1)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            return auc

        study = optuna.create_study(direction='maximize',
                                    study_name=f'{model_type}_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name='auc')
        study.optimize(objective, n_trials=self.config.get('n_trials', 50), callbacks=[mlflow_callback])
        best_params = study.best_params
        best_value = study.best_value
        logger.info(f"Best AUC: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        if model_type == 'xgboost':
            import xgboost as xgb
            final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        else:
            import lightgbm as lgb
            final_model = lgb.LGBMClassifier(**best_params, verbosity=-1)

        final_model.fit(X_train, y_train)
        self.best_model = final_model
        return {'best_params': best_params, 'best_score': best_value, 'model': final_model}

    def feature_importance_analysis(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model doesn't have feature importance")
            return pd.DataFrame()
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        total = importance_df['importance'].sum() or 1.0
        importance_df['cumulative_importance'] /= total
        logger.info("Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        return importance_df

    def save_model(self, model_name: str, filepath: str):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        model_data = self.models[model_name]
        joblib.dump({
            'model': model_data['model'],
            'metrics': model_data['metrics'],
            'training_date': model_data['training_date'],
            'config': self.config,
        }, filepath)
        logger.info(f"Saved model {model_name} to {filepath}")

    def load_model(self, filepath: str) -> Dict:
        model_data = joblib.load(filepath)
        logger.info(f"Loaded model from {filepath}")
        return model_data

    def generate_prediction_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        report: Dict[str, Any] = {}
        for model_name, model_data in self.models.items():
            model = model_data['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_test, y_pred)
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            report[model_name] = {
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': clf_report,
                'predictions': {
                    'y_pred': y_pred.tolist(),
                    'y_pred_proba': y_pred_proba.tolist(),
                },
            }
            logger.info(
                f"{model_name} Test - Acc: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, ROI: {metrics['roi']:.2f}%"
            )
        return report
