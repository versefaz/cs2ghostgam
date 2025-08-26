#!/usr/bin/env python3
"""
ML Model Trainer - Production Ready
Trains CS2 match prediction models using collected historical data
"""

import logging
import json
import pickle
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from ml_pipeline.training.data_collector import MLDataCollector
from ml_pipeline.evaluation.model_validator import ModelValidator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Production-ready ML model trainer for CS2 match prediction"""
    
    def __init__(self, data_collector: MLDataCollector = None):
        self.data_collector = data_collector or MLDataCollector()
        self.validator = ModelValidator()
        
        # Model configurations
        self.model_configs = {
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'random_state': [42]
                },
                'cv_folds': 5,
                'scoring': 'roc_auc'
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'random_state': [42]
                },
                'cv_folds': 5,
                'scoring': 'roc_auc'
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'random_state': [42],
                    'max_iter': [1000]
                },
                'cv_folds': 5,
                'scoring': 'roc_auc'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                },
                'cv_folds': 5,
                'scoring': 'roc_auc'
            }
        }
        
        # Best model storage
        self.best_model = None
        self.best_score = 0.0
        self.best_model_name = None
        self.feature_names = []
        self.scaler = None
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare training data"""
        logger.info("Preparing training data...")
        
        # Get training data
        X, y, feature_names = self.data_collector.get_training_data()
        
        if len(X) == 0:
            logger.error("No training data available")
            raise ValueError("No training data found")
        
        # Data validation
        if len(X) < 100:
            logger.warning(f"Limited training data: {len(X)} samples")
        
        # Check for class imbalance
        class_counts = np.bincount(y)
        class_ratio = min(class_counts) / max(class_counts)
        
        if class_ratio < 0.3:
            logger.warning(f"Class imbalance detected: {class_counts}, ratio: {class_ratio:.2f}")
        
        # Handle missing values
        if np.isnan(X).any():
            logger.info("Handling missing values...")
            X = np.nan_to_num(X, nan=0.0)
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = feature_names
        
        logger.info(f"Data prepared: {len(X)} samples, {len(feature_names)} features")
        logger.info(f"Class distribution: {dict(zip(['Team2 wins', 'Team1 wins'], class_counts))}")
        
        return X_scaled, y, feature_names
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train a specific model with hyperparameter tuning"""
        logger.info(f"Training {model_name} model...")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        
        # Create model pipeline
        if model_name == 'logistic_regression':
            # Logistic regression benefits from scaling
            pipeline = Pipeline([
                ('classifier', config['model']())
            ])
            param_grid = {f'classifier__{k}': v for k, v in config['params'].items()}
        else:
            pipeline = config['model']()
            param_grid = config['params']
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=config['cv_folds'],
            scoring=config['scoring'],
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X, y)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            best_model, X, y, 
            cv=config['cv_folds'], 
            scoring=config['scoring']
        )
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'best_cv_score': best_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_name': model_name
        }
        
        logger.info(f"{model_name} - Best CV Score: {best_score:.4f} (±{cv_scores.std()*2:.4f})")
        
        return results
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Train all configured models and find the best one"""
        logger.info("Training all models...")
        
        # Prepare data
        X, y, feature_names = self.prepare_data()
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Train each model
        for model_name in self.model_configs.keys():
            try:
                model_results = self.train_model(model_name, X_train, y_train)
                
                # Evaluate on test set
                test_predictions = model_results['model'].predict(X_test)
                test_probabilities = model_results['model'].predict_proba(X_test)[:, 1]
                
                # Calculate test metrics
                test_metrics = {
                    'accuracy': accuracy_score(y_test, test_predictions),
                    'precision': precision_score(y_test, test_predictions),
                    'recall': recall_score(y_test, test_predictions),
                    'f1': f1_score(y_test, test_predictions),
                    'roc_auc': roc_auc_score(y_test, test_probabilities)
                }
                
                model_results['test_metrics'] = test_metrics
                model_results['test_size'] = len(X_test)
                
                results[model_name] = model_results
                
                # Track best model
                if test_metrics['roc_auc'] > self.best_score:
                    self.best_score = test_metrics['roc_auc']
                    self.best_model = model_results['model']
                    self.best_model_name = model_name
                
                logger.info(f"{model_name} test metrics: "
                          f"Accuracy={test_metrics['accuracy']:.3f}, "
                          f"ROC-AUC={test_metrics['roc_auc']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        if self.best_model is None:
            raise RuntimeError("No models were successfully trained")
        
        logger.info(f"Best model: {self.best_model_name} (ROC-AUC: {self.best_score:.4f})")
        
        return results
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """Get feature importance from the best model"""
        model = self.best_model
        if model_name and model_name in self.model_configs:
            # Get specific model if requested
            pass
        
        if model is None:
            logger.warning("No trained model available")
            return {}
        
        # Extract feature importance
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
            else:
                logger.warning("Model doesn't support feature importance")
                return {}
            
            # Create importance dictionary
            for i, importance in enumerate(importances):
                if i < len(self.feature_names):
                    importance_dict[self.feature_names[i]] = float(importance)
            
            # Sort by importance
            importance_dict = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
        
        return importance_dict
    
    def save_model(self, model_path: str = "models/cs2_prediction_model.pkl"):
        """Save the best trained model"""
        if self.best_model is None:
            logger.error("No trained model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare model package
            model_package = {
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_name': self.best_model_name,
                'best_score': self.best_score,
                'training_date': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
            
            # Save using joblib for sklearn models
            joblib.dump(model_package, model_path)
            
            logger.info(f"Model saved to {model_path}")
            
            # Save model metadata
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            metadata = {
                'model_name': self.best_model_name,
                'best_score': self.best_score,
                'feature_count': len(self.feature_names),
                'training_date': datetime.utcnow().isoformat(),
                'feature_importance': self.get_feature_importance()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model metadata saved to {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str = "models/cs2_prediction_model.pkl") -> bool:
        """Load a trained model"""
        try:
            model_package = joblib.load(model_path)
            
            self.best_model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.best_model_name = model_package['model_name']
            self.best_score = model_package['best_score']
            
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model: {self.best_model_name}, Score: {self.best_score:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Make prediction using the trained model"""
        if self.best_model is None:
            raise RuntimeError("No trained model available")
        
        if self.scaler is None:
            raise RuntimeError("No scaler available")
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(features_scaled)[0]
        
        # Return prediction and confidence
        confidence = max(probability)
        
        return int(prediction), float(confidence)
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if self.best_model is None:
            raise RuntimeError("No trained model available")
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        predictions = self.best_model.predict(X_test_scaled)
        probabilities = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'roc_auc': roc_auc_score(y_test, probabilities)
        }
        
        return metrics
    
    def generate_training_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive training report"""
        report = []
        report.append("=" * 60)
        report.append("CS2 MATCH PREDICTION MODEL TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"Training Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of Models Trained: {len(results)}")
        report.append("")
        
        # Data statistics
        stats = self.data_collector.get_data_statistics()
        report.append("TRAINING DATA STATISTICS:")
        report.append(f"  Total Matches: {stats['total_matches']}")
        report.append(f"  Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        report.append(f"  Team1 Win Rate: {stats['overall_win_rate']['team1_wins']:.2%}")
        report.append("")
        
        # Model results
        report.append("MODEL PERFORMANCE COMPARISON:")
        report.append("-" * 40)
        
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            report.append(f"{model_name.upper()}:")
            report.append(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']*2:.4f})")
            report.append(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            report.append(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
            report.append(f"  Test F1: {test_metrics['f1']:.4f}")
            report.append("")
        
        # Best model details
        report.append("BEST MODEL:")
        report.append(f"  Model: {self.best_model_name}")
        report.append(f"  ROC-AUC Score: {self.best_score:.4f}")
        report.append("")
        
        # Feature importance
        importance = self.get_feature_importance()
        if importance:
            report.append("TOP 10 FEATURE IMPORTANCE:")
            for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                report.append(f"  {i+1:2d}. {feature}: {imp:.4f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


# Usage example and main training script
async def main():
    """Main training pipeline"""
    logger.info("Starting CS2 model training pipeline...")
    
    # Initialize components
    data_collector = MLDataCollector()
    trainer = ModelTrainer(data_collector)
    
    # Collect training data
    logger.info("Collecting training data...")
    await data_collector.collect_historical_data(days_back=90, max_matches=1000)
    
    # Train models
    logger.info("Training models...")
    results = trainer.train_all_models()
    
    # Save best model
    model_saved = trainer.save_model("models/cs2_simple_model.pkl")
    
    if model_saved:
        logger.info("Model training completed successfully!")
        
        # Generate and save report
        report = trainer.generate_training_report(results)
        print(report)
        
        with open("models/training_report.txt", "w") as f:
            f.write(report)
        
        # Test model loading
        logger.info("Testing model loading...")
        test_trainer = ModelTrainer()
        if test_trainer.load_model("models/cs2_simple_model.pkl"):
            logger.info("Model loading test successful!")
        else:
            logger.error("Model loading test failed!")
    
    else:
        logger.error("Failed to save model!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
