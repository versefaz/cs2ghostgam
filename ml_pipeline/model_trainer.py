"""
CS2 Model Trainer - XGBoost/LogisticRegression Training with Validation
"""

import os
import json
import pickle
import hashlib
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

# ML imports with fallback
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: str = 'xgboost'  # 'xgboost' or 'logistic'
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    hyperparameter_tuning: bool = True
    mlflow_tracking: bool = False  # Disabled by default
    mlflow_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "cs2_betting_model"
    model_save_path: str = "./models"
    
    # Model-specific parameters
    xgb_params: Dict = None
    logistic_params: Dict = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
        
        if self.logistic_params is None:
            self.logistic_params = {
                'max_iter': 1000,
                'solver': 'lbfgs',
                'C': 1.0
            }

class CS2ModelTrainer:
    """
    Model trainer for CS2 betting predictions with comprehensive validation
    Supports XGBoost and Logistic Regression with hyperparameter tuning
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.best_params = {}
        self.cv_scores = []
        self.feature_importance = {}
        self.model_version = None
        self.model_hash = None
        self.training_history = {}
        
        # Setup MLflow if available and enabled
        if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self.mlflow_client = MlflowClient()
        
        # Ensure model save directory exists
        os.makedirs(self.config.model_save_path, exist_ok=True)
        
        logger.info(f"CS2ModelTrainer initialized with {self.config.model_type} model")
    
    def get_model(self, params: Dict = None):
        """Get model instance with parameters"""
        if self.config.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            
            model_params = self.config.xgb_params.copy()
            model_params['random_state'] = self.config.random_state
            if params:
                model_params.update(params)
            return xgb.XGBClassifier(**model_params)
        
        elif self.config.model_type == 'logistic':
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
            
            model_params = self.config.logistic_params.copy()
            model_params['random_state'] = self.config.random_state
            if params:
                model_params.update(params)
            return LogisticRegression(**model_params)
        
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def get_hyperparameter_grid(self) -> Dict:
        """Get hyperparameter grid for tuning"""
        if self.config.model_type == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif self.config.model_type == 'logistic':
            return {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        else:
            return {}
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train the model with comprehensive validation
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features for importance analysis
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training {self.config.model_type} model on {X.shape[0]} samples with {X.shape[1]} features")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model training")
        
        # Validate input data
        self._validate_training_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Start MLflow run if enabled
        if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
            mlflow.start_run()
            mlflow.log_params({
                'model_type': self.config.model_type,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X.shape[1],
                'cv_folds': self.config.cv_folds
            })
        
        try:
            # Hyperparameter tuning
            if self.config.hyperparameter_tuning:
                logger.info("Performing hyperparameter tuning...")
                self.model, self.best_params = self._tune_hyperparameters(X_train, y_train)
            else:
                self.model = self.get_model()
                self.model.fit(X_train, y_train)
                self.best_params = self.model.get_params()
            
            # Cross-validation
            logger.info("Performing cross-validation...")
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            self.cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='roc_auc')
            
            # Evaluate on test set
            test_results = self._evaluate_model(X_test, y_test)
            
            # Feature importance analysis
            if hasattr(self.model, 'feature_importances_'):
                self._analyze_feature_importance(feature_names)
            
            # Generate model metadata
            self.model_hash = self._generate_model_hash()
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store training history
            self.training_history = {
                'timestamp': datetime.now().isoformat(),
                'model_type': self.config.model_type,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X.shape[1],
                'cv_mean': np.mean(self.cv_scores),
                'cv_std': np.std(self.cv_scores),
                'test_auc': test_results['roc_auc']
            }
            
            # Log to MLflow
            if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                self._log_to_mlflow(test_results)
            
            # Save model and metadata
            self._save_model(feature_names)
            
            # Compile results
            results = {
                'cv_scores': self.cv_scores,
                'cv_mean': np.mean(self.cv_scores),
                'cv_std': np.std(self.cv_scores),
                'test_results': test_results,
                'best_params': self.best_params,
                'feature_importance': self.feature_importance,
                'model_version': self.model_version,
                'model_hash': self.model_hash,
                'model_path': self._get_model_path()
            }
            
            logger.info(f"Training completed successfully!")
            logger.info(f"CV AUC: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            logger.info(f"Test AUC: {test_results['roc_auc']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                mlflow.end_run()
    
    def _validate_training_data(self, X: np.ndarray, y: np.ndarray):
        """Validate training data"""
        if X.shape[0] != len(y):
            raise ValueError(f"Feature matrix and labels have different lengths: {X.shape[0]} vs {len(y)}")
        
        if X.shape[0] < 50:
            logger.warning(f"Very small dataset: {X.shape[0]} samples. Results may be unreliable.")
        
        if len(np.unique(y)) != 2:
            raise ValueError(f"Expected binary classification, got {len(np.unique(y))} classes")
        
        # Check for missing values
        if np.isnan(X).any():
            raise ValueError("Feature matrix contains NaN values")
        
        # Check class balance
        class_balance = np.mean(y)
        if class_balance < 0.1 or class_balance > 0.9:
            logger.warning(f"Severe class imbalance: {class_balance:.2%} positive class")
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Perform hyperparameter tuning using GridSearchCV"""
        base_model = self.get_model()
        param_grid = self.get_hyperparameter_grid()
        
        if not param_grid:
            logger.info("No hyperparameter grid defined, using default parameters")
            base_model.fit(X, y)
            return base_model, base_model.get_params()
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=self.config.cv_folds,
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info("Test Set Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def _analyze_feature_importance(self, feature_names: List[str] = None):
        """Analyze and store feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if feature_names and len(feature_names) == len(importances):
                self.feature_importance = dict(zip(feature_names, importances))
            else:
                self.feature_importance = {f'feature_{i}': imp for i, imp in enumerate(importances)}
            
            # Log top 10 most important features
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                logger.info(f"  {i+1}. {feature}: {importance:.4f}")
    
    def _generate_model_hash(self) -> str:
        """Generate hash for model versioning"""
        model_str = str(self.best_params) + str(self.config.model_type)
        return hashlib.md5(model_str.encode()).hexdigest()[:8]
    
    def _log_to_mlflow(self, test_results: Dict):
        """Log metrics and model to MLflow"""
        # Log metrics
        mlflow.log_metrics(test_results)
        mlflow.log_metrics({
            'cv_mean': np.mean(self.cv_scores),
            'cv_std': np.std(self.cv_scores)
        })
        
        # Log parameters
        mlflow.log_params(self.best_params)
        
        # Log model
        if self.config.model_type == 'xgboost':
            mlflow.xgboost.log_model(self.model, "model")
        else:
            mlflow.sklearn.log_model(self.model, "model")
    
    def _get_model_path(self) -> str:
        """Get model file path"""
        filename = f"cs2_model_{self.config.model_type}_{self.model_version}_{self.model_hash}.pkl"
        return os.path.join(self.config.model_save_path, filename)
    
    def _save_model(self, feature_names: List[str] = None):
        """Save model and metadata to disk"""
        model_path = self._get_model_path()
        
        # Prepare model package
        model_package = {
            'model': self.model,
            'config': self.config,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'feature_names': feature_names,
            'model_version': self.model_version,
            'model_hash': self.model_hash,
            'training_history': self.training_history,
            'cv_scores': self.cv_scores
        }
        
        # Save model package
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save metadata as JSON
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        metadata = {
            'model_type': self.config.model_type,
            'model_version': self.model_version,
            'model_hash': self.model_hash,
            'training_history': self.training_history,
            'best_params': self.best_params,
            'feature_names': feature_names,
            'cv_mean': float(np.mean(self.cv_scores)),
            'cv_std': float(np.std(self.cv_scores))
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load trained model from disk
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Model package dictionary
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        # Restore model state
        self.model = model_package['model']
        self.config = model_package['config']
        self.best_params = model_package['best_params']
        self.feature_importance = model_package['feature_importance']
        self.model_version = model_package['model_version']
        self.model_hash = model_package['model_hash']
        self.training_history = model_package['training_history']
        self.cv_scores = model_package['cv_scores']
        
        logger.info(f"Model loaded from: {model_path}")
        logger.info(f"Model version: {self.model_version}")
        
        return model_package
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with loaded model"""
        if self.model is None:
            raise ValueError("No model loaded. Train a model or load from disk first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("No model loaded. Train a model or load from disk first.")
        
        return self.model.predict_proba(X)
    
    def validate_model(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Validate model on separate validation set
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Train a model or load from disk first.")
        
        logger.info(f"Validating model on {len(X_val)} samples")
        
        # Make predictions
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        validation_results = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        logger.info("Validation Results:")
        for metric, value in validation_results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return validation_results
