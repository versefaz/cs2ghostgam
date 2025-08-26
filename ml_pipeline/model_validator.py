"""
CS2 Model Validator - Model Loading Validation and Smoke Testing
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class CS2ModelValidator:
    """
    Comprehensive model validation and smoke testing for CS2 betting models
    Ensures models can be loaded correctly and perform inference
    """
    
    def __init__(self):
        self.validation_results = {}
        self.smoke_test_results = {}
        logger.info("CS2ModelValidator initialized")
    
    def validate_model_file(self, model_path: str) -> Dict[str, Any]:
        """
        Validate that model file exists and can be loaded
        
        Args:
            model_path: Path to model file
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating model file: {model_path}")
        
        results = {
            'file_exists': False,
            'file_size_mb': 0,
            'loadable': False,
            'model_type': None,
            'model_version': None,
            'feature_count': 0,
            'has_metadata': False,
            'error': None
        }
        
        try:
            # Check file existence
            if not os.path.exists(model_path):
                results['error'] = f"Model file not found: {model_path}"
                return results
            
            results['file_exists'] = True
            results['file_size_mb'] = round(os.path.getsize(model_path) / (1024 * 1024), 2)
            
            # Try to load model
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            results['loadable'] = True
            
            # Extract model information
            if 'model' in model_package:
                model = model_package['model']
                results['model_type'] = type(model).__name__
                
                # Check for feature count
                if hasattr(model, 'n_features_in_'):
                    results['feature_count'] = model.n_features_in_
                elif 'feature_names' in model_package and model_package['feature_names']:
                    results['feature_count'] = len(model_package['feature_names'])
            
            # Extract metadata
            if 'model_version' in model_package:
                results['model_version'] = model_package['model_version']
            
            # Check for metadata file
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            results['has_metadata'] = os.path.exists(metadata_path)
            
            logger.info(f"Model validation successful: {results['model_type']} with {results['feature_count']} features")
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Model validation failed: {str(e)}")
        
        self.validation_results[model_path] = results
        return results
    
    def smoke_test_inference(self, model_path: str, sample_size: int = 10) -> Dict[str, Any]:
        """
        Perform smoke test for model inference
        
        Args:
            model_path: Path to model file
            sample_size: Number of synthetic samples to test
            
        Returns:
            Smoke test results
        """
        logger.info(f"Running smoke test inference for: {model_path}")
        
        results = {
            'inference_successful': False,
            'prediction_shape': None,
            'probability_shape': None,
            'prediction_range': None,
            'probability_range': None,
            'execution_time_ms': 0,
            'error': None
        }
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            model = model_package['model']
            feature_count = results.get('feature_count', 100)  # Default fallback
            
            # Determine feature count
            if hasattr(model, 'n_features_in_'):
                feature_count = model.n_features_in_
            elif 'feature_names' in model_package and model_package['feature_names']:
                feature_count = len(model_package['feature_names'])
            
            # Generate synthetic test data
            X_test = self._generate_synthetic_data(sample_size, feature_count)
            
            # Time the inference
            start_time = datetime.now()
            
            # Test predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            # Analyze results
            results['inference_successful'] = True
            results['prediction_shape'] = y_pred.shape
            results['probability_shape'] = y_pred_proba.shape
            results['prediction_range'] = (int(y_pred.min()), int(y_pred.max()))
            results['probability_range'] = (float(y_pred_proba.min()), float(y_pred_proba.max()))
            results['execution_time_ms'] = round(execution_time, 2)
            
            # Validate prediction outputs
            self._validate_predictions(y_pred, y_pred_proba, results)
            
            logger.info(f"Smoke test successful - Execution time: {execution_time:.2f}ms")
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Smoke test failed: {str(e)}")
        
        self.smoke_test_results[model_path] = results
        return results
    
    def _generate_synthetic_data(self, sample_size: int, feature_count: int) -> np.ndarray:
        """Generate synthetic test data for smoke testing"""
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic CS2 feature ranges
        synthetic_data = []
        
        for _ in range(sample_size):
            sample = []
            
            for feature_idx in range(feature_count):
                # Generate different types of features based on common CS2 patterns
                if feature_idx % 10 == 0:  # Win rates (0-1)
                    value = np.random.uniform(0.2, 0.8)
                elif feature_idx % 10 == 1:  # K/D ratios (0.5-2.0)
                    value = np.random.uniform(0.5, 2.0)
                elif feature_idx % 10 == 2:  # ADR (30-100)
                    value = np.random.uniform(30, 100)
                elif feature_idx % 10 == 3:  # Ratings (0.5-1.5)
                    value = np.random.uniform(0.5, 1.5)
                elif feature_idx % 10 == 4:  # Rounds (0-30)
                    value = np.random.uniform(0, 30)
                elif feature_idx % 10 == 5:  # Rankings (1-50)
                    value = np.random.uniform(1, 50)
                elif feature_idx % 10 == 6:  # Days since last match (0-30)
                    value = np.random.uniform(0, 30)
                elif feature_idx % 10 == 7:  # Tournament tier (1-5)
                    value = np.random.uniform(1, 5)
                elif feature_idx % 10 == 8:  # Binary features (0 or 1)
                    value = np.random.choice([0, 1])
                else:  # General features (-2 to 2)
                    value = np.random.uniform(-2, 2)
                
                sample.append(value)
            
            synthetic_data.append(sample)
        
        return np.array(synthetic_data)
    
    def _validate_predictions(self, y_pred: np.ndarray, y_pred_proba: np.ndarray, results: Dict):
        """Validate prediction outputs are reasonable"""
        # Check prediction values are binary (0 or 1)
        unique_preds = np.unique(y_pred)
        if not all(pred in [0, 1] for pred in unique_preds):
            results['error'] = f"Invalid prediction values: {unique_preds}"
            return
        
        # Check probabilities are in valid range [0, 1]
        if y_pred_proba.min() < 0 or y_pred_proba.max() > 1:
            results['error'] = f"Invalid probability range: [{y_pred_proba.min()}, {y_pred_proba.max()}]"
            return
        
        # Check probability matrix has 2 columns (binary classification)
        if y_pred_proba.shape[1] != 2:
            results['error'] = f"Expected 2 probability columns, got {y_pred_proba.shape[1]}"
            return
        
        # Check probabilities sum to 1 (approximately)
        prob_sums = y_pred_proba.sum(axis=1)
        if not np.allclose(prob_sums, 1.0, atol=1e-6):
            results['error'] = f"Probabilities don't sum to 1: {prob_sums[:5]}"
            return
        
        logger.info("Prediction validation passed")
    
    def comprehensive_model_test(self, model_path: str) -> Dict[str, Any]:
        """
        Run comprehensive model testing including validation and smoke tests
        
        Args:
            model_path: Path to model file
            
        Returns:
            Complete test results
        """
        logger.info(f"Running comprehensive model test: {model_path}")
        
        # File validation
        validation_results = self.validate_model_file(model_path)
        
        # Smoke test (only if validation passed)
        smoke_results = {}
        if validation_results['loadable']:
            smoke_results = self.smoke_test_inference(model_path)
        else:
            logger.warning("Skipping smoke test due to validation failure")
        
        # Compile comprehensive results
        comprehensive_results = {
            'model_path': model_path,
            'test_timestamp': datetime.now().isoformat(),
            'validation': validation_results,
            'smoke_test': smoke_results,
            'overall_status': 'PASS' if validation_results['loadable'] and smoke_results.get('inference_successful', False) else 'FAIL'
        }
        
        logger.info(f"Comprehensive test status: {comprehensive_results['overall_status']}")
        
        return comprehensive_results
    
    def test_all_models_in_directory(self, models_dir: str) -> Dict[str, Any]:
        """
        Test all model files in a directory
        
        Args:
            models_dir: Directory containing model files
            
        Returns:
            Results for all models
        """
        logger.info(f"Testing all models in directory: {models_dir}")
        
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            return {}
        
        results = {}
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            logger.warning(f"No model files found in {models_dir}")
            return {}
        
        logger.info(f"Found {len(model_files)} model files to test")
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            try:
                results[model_file] = self.comprehensive_model_test(model_path)
            except Exception as e:
                logger.error(f"Failed to test {model_file}: {str(e)}")
                results[model_file] = {
                    'model_path': model_path,
                    'test_timestamp': datetime.now().isoformat(),
                    'overall_status': 'ERROR',
                    'error': str(e)
                }
        
        # Summary statistics
        total_models = len(results)
        passed_models = sum(1 for r in results.values() if r.get('overall_status') == 'PASS')
        
        logger.info(f"Model testing complete: {passed_models}/{total_models} models passed")
        
        return {
            'summary': {
                'total_models': total_models,
                'passed_models': passed_models,
                'failed_models': total_models - passed_models,
                'success_rate': round(passed_models / total_models * 100, 1) if total_models > 0 else 0
            },
            'results': results
        }
    
    def generate_test_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate a detailed test report
        
        Args:
            results: Test results dictionary
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("CS2 MODEL VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if 'summary' in results:
            summary = results['summary']
            report_lines.append("SUMMARY:")
            report_lines.append(f"  Total Models: {summary['total_models']}")
            report_lines.append(f"  Passed: {summary['passed_models']}")
            report_lines.append(f"  Failed: {summary['failed_models']}")
            report_lines.append(f"  Success Rate: {summary['success_rate']}%")
            report_lines.append("")
        
        if 'results' in results:
            report_lines.append("DETAILED RESULTS:")
            report_lines.append("-" * 40)
            
            for model_name, model_results in results['results'].items():
                report_lines.append(f"\nModel: {model_name}")
                report_lines.append(f"Status: {model_results.get('overall_status', 'UNKNOWN')}")
                
                if 'validation' in model_results:
                    val = model_results['validation']
                    report_lines.append(f"  File Size: {val.get('file_size_mb', 0)} MB")
                    report_lines.append(f"  Model Type: {val.get('model_type', 'Unknown')}")
                    report_lines.append(f"  Features: {val.get('feature_count', 0)}")
                    report_lines.append(f"  Version: {val.get('model_version', 'Unknown')}")
                
                if 'smoke_test' in model_results:
                    smoke = model_results['smoke_test']
                    if smoke.get('inference_successful'):
                        report_lines.append(f"  Inference Time: {smoke.get('execution_time_ms', 0)} ms")
                        report_lines.append(f"  Prediction Range: {smoke.get('prediction_range', 'Unknown')}")
                
                if model_results.get('error'):
                    report_lines.append(f"  Error: {model_results['error']}")
        
        report_lines.append("\n" + "=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Test report saved to: {output_path}")
        
        return report_text

def run_model_validation_suite(models_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to run complete model validation suite
    
    Args:
        models_dir: Directory containing models to test
        output_dir: Optional directory to save reports
        
    Returns:
        Complete validation results
    """
    validator = CS2ModelValidator()
    
    # Test all models
    results = validator.test_all_models_in_directory(models_dir)
    
    # Generate report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"model_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        validator.generate_test_report(results, report_path)
    
    return results
