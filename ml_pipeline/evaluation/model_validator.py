#!/usr/bin/env python3
"""
Model Validator - ML Pipeline Component
Validates trained models for performance, stability, and production readiness
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Model validation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_mean: float
    cross_val_std: float
    feature_importance_stability: float
    prediction_consistency: float


@dataclass
class ValidationResult:
    """Complete validation result"""
    is_valid: bool
    metrics: ValidationMetrics
    issues: List[str]
    recommendations: List[str]
    confidence_score: float


class ModelValidator:
    """Validates ML models for production readiness"""
    
    def __init__(self, 
                 min_accuracy: float = 0.65,
                 min_precision: float = 0.60,
                 min_recall: float = 0.60,
                 min_f1: float = 0.60,
                 min_roc_auc: float = 0.65,
                 max_cv_std: float = 0.10):
        """
        Initialize model validator with thresholds
        
        Args:
            min_accuracy: Minimum required accuracy
            min_precision: Minimum required precision
            min_recall: Minimum required recall
            min_f1: Minimum required F1 score
            min_roc_auc: Minimum required ROC AUC
            max_cv_std: Maximum allowed cross-validation standard deviation
        """
        self.min_accuracy = min_accuracy
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_f1 = min_f1
        self.min_roc_auc = min_roc_auc
        self.max_cv_std = max_cv_std
        
        logger.info("Model validator initialized with production thresholds")
    
    def validate_model(self, 
                      model: Any, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      X_train: Optional[np.ndarray] = None,
                      y_train: Optional[np.ndarray] = None) -> ValidationResult:
        """
        Comprehensive model validation
        
        Args:
            model: Trained model to validate
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional, for cross-validation)
            y_train: Training labels (optional, for cross-validation)
            
        Returns:
            ValidationResult with metrics and recommendations
        """
        try:
            logger.info("Starting comprehensive model validation...")
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # Try to get prediction probabilities
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                logger.warning("Model doesn't support predict_proba, skipping ROC AUC")
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC (if probabilities available)
            roc_auc = 0.0
            if y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    logger.warning("Could not calculate ROC AUC")
            
            # Cross-validation (if training data provided)
            cv_mean, cv_std = 0.0, 0.0
            if X_train is not None and y_train is not None:
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as e:
                    logger.warning(f"Cross-validation failed: {e}")
            
            # Feature importance stability (if available)
            feature_stability = self._check_feature_importance_stability(model)
            
            # Prediction consistency
            prediction_consistency = self._check_prediction_consistency(model, X_test)
            
            # Create metrics object
            metrics = ValidationMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                cross_val_mean=cv_mean,
                cross_val_std=cv_std,
                feature_importance_stability=feature_stability,
                prediction_consistency=prediction_consistency
            )
            
            # Validate against thresholds
            is_valid, issues, recommendations = self._validate_against_thresholds(metrics)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(metrics)
            
            result = ValidationResult(
                is_valid=is_valid,
                metrics=metrics,
                issues=issues,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
            logger.info(f"Model validation completed. Valid: {is_valid}, Confidence: {confidence_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                metrics=ValidationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0),
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Fix validation errors before deployment"],
                confidence_score=0.0
            )
    
    def _check_feature_importance_stability(self, model: Any) -> float:
        """Check if model has stable feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Simple stability check: ensure no single feature dominates
                max_importance = np.max(importances)
                if max_importance > 0.8:  # Single feature dominates
                    return 0.2
                elif max_importance > 0.6:  # High concentration
                    return 0.5
                else:
                    return 0.9  # Good distribution
            return 0.5  # Unknown, assume moderate
        except:
            return 0.5
    
    def _check_prediction_consistency(self, model: Any, X_test: np.ndarray) -> float:
        """Check prediction consistency with small perturbations"""
        try:
            if len(X_test) < 10:
                return 0.5
            
            # Take a small sample
            sample_size = min(50, len(X_test))
            sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[sample_idx]
            
            # Original predictions
            pred_original = model.predict(X_sample)
            
            # Add small noise and predict again
            noise_level = 0.01 * np.std(X_sample, axis=0)
            X_noisy = X_sample + np.random.normal(0, noise_level, X_sample.shape)
            pred_noisy = model.predict(X_noisy)
            
            # Calculate consistency
            consistency = np.mean(pred_original == pred_noisy)
            return consistency
            
        except:
            return 0.5
    
    def _validate_against_thresholds(self, metrics: ValidationMetrics) -> Tuple[bool, List[str], List[str]]:
        """Validate metrics against thresholds"""
        issues = []
        recommendations = []
        
        # Check accuracy
        if metrics.accuracy < self.min_accuracy:
            issues.append(f"Accuracy {metrics.accuracy:.3f} below threshold {self.min_accuracy}")
            recommendations.append("Improve model accuracy through feature engineering or algorithm tuning")
        
        # Check precision
        if metrics.precision < self.min_precision:
            issues.append(f"Precision {metrics.precision:.3f} below threshold {self.min_precision}")
            recommendations.append("Reduce false positives to improve precision")
        
        # Check recall
        if metrics.recall < self.min_recall:
            issues.append(f"Recall {metrics.recall:.3f} below threshold {self.min_recall}")
            recommendations.append("Reduce false negatives to improve recall")
        
        # Check F1 score
        if metrics.f1_score < self.min_f1:
            issues.append(f"F1 score {metrics.f1_score:.3f} below threshold {self.min_f1}")
            recommendations.append("Balance precision and recall to improve F1 score")
        
        # Check ROC AUC
        if metrics.roc_auc > 0 and metrics.roc_auc < self.min_roc_auc:
            issues.append(f"ROC AUC {metrics.roc_auc:.3f} below threshold {self.min_roc_auc}")
            recommendations.append("Improve model's ability to distinguish between classes")
        
        # Check cross-validation stability
        if metrics.cross_val_std > self.max_cv_std:
            issues.append(f"Cross-validation std {metrics.cross_val_std:.3f} above threshold {self.max_cv_std}")
            recommendations.append("Model shows high variance, consider regularization or more data")
        
        # Check feature importance stability
        if metrics.feature_importance_stability < 0.3:
            issues.append("Poor feature importance stability detected")
            recommendations.append("Review feature selection and model complexity")
        
        # Check prediction consistency
        if metrics.prediction_consistency < 0.7:
            issues.append("Low prediction consistency with input perturbations")
            recommendations.append("Model may be overfitting, consider regularization")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            recommendations.append("Model passes all validation checks and is ready for production")
        
        return is_valid, issues, recommendations
    
    def _calculate_confidence_score(self, metrics: ValidationMetrics) -> float:
        """Calculate overall confidence score for the model"""
        scores = []
        
        # Accuracy contribution
        scores.append(min(metrics.accuracy / self.min_accuracy, 1.0) * 0.25)
        
        # Precision contribution
        scores.append(min(metrics.precision / self.min_precision, 1.0) * 0.20)
        
        # Recall contribution
        scores.append(min(metrics.recall / self.min_recall, 1.0) * 0.20)
        
        # F1 contribution
        scores.append(min(metrics.f1_score / self.min_f1, 1.0) * 0.15)
        
        # ROC AUC contribution (if available)
        if metrics.roc_auc > 0:
            scores.append(min(metrics.roc_auc / self.min_roc_auc, 1.0) * 0.10)
        else:
            scores.append(0.05)  # Partial credit
        
        # Stability contributions
        scores.append(metrics.feature_importance_stability * 0.05)
        scores.append(metrics.prediction_consistency * 0.05)
        
        return sum(scores)
    
    def save_validation_report(self, result: ValidationResult, filepath: str):
        """Save validation report to file"""
        try:
            report = {
                'validation_result': {
                    'is_valid': result.is_valid,
                    'confidence_score': result.confidence_score,
                    'timestamp': pd.Timestamp.now().isoformat()
                },
                'metrics': {
                    'accuracy': result.metrics.accuracy,
                    'precision': result.metrics.precision,
                    'recall': result.metrics.recall,
                    'f1_score': result.metrics.f1_score,
                    'roc_auc': result.metrics.roc_auc,
                    'cross_val_mean': result.metrics.cross_val_mean,
                    'cross_val_std': result.metrics.cross_val_std,
                    'feature_importance_stability': result.metrics.feature_importance_stability,
                    'prediction_consistency': result.metrics.prediction_consistency
                },
                'issues': result.issues,
                'recommendations': result.recommendations
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Validation report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
