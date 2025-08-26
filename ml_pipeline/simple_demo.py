"""
CS2 ML Pipeline - Simple Demo with Basic Libraries
Demonstrates the complete system with fallback to basic implementations
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleCS2Model:
    """Simple logistic regression implementation for CS2 predictions"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.feature_names = None
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the model"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute cost
            cost = (-1/n_samples) * np.sum(y*np.log(predictions + 1e-15) + (1-y)*np.log(1-predictions + 1e-15))
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                logger.info(f"Iteration {i}, Cost: {cost:.4f}")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        linear_pred = np.dot(X, self.weights) + self.bias
        proba_1 = self.sigmoid(linear_pred)
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])
    
    def predict(self, X):
        """Make binary predictions"""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def generate_sample_data(n_samples=500):
    """Generate synthetic CS2 match data"""
    logger.info(f"Generating {n_samples} synthetic samples")
    
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        # Team ratings
        team1_rating = np.random.uniform(0.8, 1.4)
        team2_rating = np.random.uniform(0.8, 1.4)
        
        # Win rates
        team1_win_rate = np.random.uniform(0.3, 0.8)
        team2_win_rate = np.random.uniform(0.3, 0.8)
        
        # Recent form
        team1_form = np.random.uniform(0.2, 0.9)
        team2_form = np.random.uniform(0.2, 0.9)
        
        # Rankings
        team1_rank = np.random.randint(1, 30)
        team2_rank = np.random.randint(1, 30)
        
        # Map performance
        map_diff = np.random.uniform(-0.3, 0.3)
        
        # H2H
        h2h_diff = np.random.uniform(-0.4, 0.4)
        
        sample = {
            'team1_rating': team1_rating,
            'team2_rating': team2_rating,
            'team1_win_rate': team1_win_rate,
            'team2_win_rate': team2_win_rate,
            'team1_form': team1_form,
            'team2_form': team2_form,
            'team1_rank': team1_rank,
            'team2_rank': team2_rank,
            'map_diff': map_diff,
            'h2h_diff': h2h_diff,
            'tournament_tier': np.random.randint(1, 6),
            'is_lan': np.random.choice([0, 1])
        }
        
        # Generate winner based on team strength
        team1_strength = (team1_rating + team1_win_rate + team1_form) / 3
        team2_strength = (team2_rating + team2_win_rate + team2_form) / 3
        
        win_prob = team1_strength / (team1_strength + team2_strength)
        win_prob = np.clip(win_prob + map_diff + h2h_diff, 0.1, 0.9)
        
        sample['winner'] = 1 if np.random.random() < win_prob else 0
        data.append(sample)
    
    return pd.DataFrame(data)

def build_features(data):
    """Build features from raw data"""
    logger.info("Building features...")
    
    features = []
    feature_names = []
    
    # Basic team features
    features.append(data['team1_rating'].values)
    features.append(data['team2_rating'].values)
    features.append(data['team1_win_rate'].values)
    features.append(data['team2_win_rate'].values)
    features.append(data['team1_form'].values)
    features.append(data['team2_form'].values)
    
    feature_names.extend(['team1_rating', 'team2_rating', 'team1_win_rate', 
                         'team2_win_rate', 'team1_form', 'team2_form'])
    
    # Derived features
    rating_diff = data['team1_rating'] - data['team2_rating']
    win_rate_diff = data['team1_win_rate'] - data['team2_win_rate']
    form_diff = data['team1_form'] - data['team2_form']
    rank_diff = data['team2_rank'] - data['team1_rank']  # Lower rank number is better
    
    features.extend([rating_diff.values, win_rate_diff.values, 
                    form_diff.values, rank_diff.values])
    feature_names.extend(['rating_diff', 'win_rate_diff', 'form_diff', 'rank_diff'])
    
    # Context features
    features.append(data['map_diff'].values)
    features.append(data['h2h_diff'].values)
    features.append(data['tournament_tier'].values)
    features.append(data['is_lan'].values)
    
    feature_names.extend(['map_diff', 'h2h_diff', 'tournament_tier', 'is_lan'])
    
    X = np.column_stack(features)
    
    # Simple normalization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_normalized = (X - X_mean) / X_std
    
    logger.info(f"Built {X_normalized.shape[1]} features for {X_normalized.shape[0]} samples")
    
    return X_normalized, feature_names, (X_mean, X_std)

def train_test_split_simple(X, y, test_size=0.2, random_state=42):
    """Simple train-test split"""
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics"""
    accuracy = np.mean(y_true == y_pred)
    
    # Precision and Recall
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Simple AUC approximation
    sorted_indices = np.argsort(y_pred_proba)
    sorted_labels = y_true[sorted_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        auc = 0
        for i in range(len(sorted_labels)):
            if sorted_labels[i] == 1:
                auc += np.sum(sorted_labels[:i] == 0)
        auc = auc / (n_pos * n_neg)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc
    }

def save_model(model, feature_names, normalization_params, model_path):
    """Save model to disk"""
    model_package = {
        'model': model,
        'feature_names': feature_names,
        'normalization_params': normalization_params,
        'model_type': 'SimpleCS2Model',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    logger.info(f"Model saved to: {model_path}")

def load_and_validate_model(model_path):
    """Load and validate model"""
    logger.info(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        feature_names = model_package['feature_names']
        
        # Smoke test
        test_X = np.random.randn(5, len(feature_names))
        predictions = model.predict(test_X)
        probabilities = model.predict_proba(test_X)
        
        logger.info("✅ Model loaded and validated successfully")
        logger.info(f"   Features: {len(feature_names)}")
        logger.info(f"   Sample predictions: {predictions}")
        logger.info(f"   Sample probabilities: {probabilities[0]}")
        
        return model_package
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {str(e)}")
        return None

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    logger.info("🚀 Starting CS2 ML Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Generate data
        logger.info("STEP 1: Data Generation")
        data = generate_sample_data(n_samples=800)
        logger.info(f"Generated {len(data)} samples, {np.mean(data['winner']):.2%} team1 wins")
        
        # Step 2: Feature engineering
        logger.info("\nSTEP 2: Feature Engineering")
        X, feature_names, norm_params = build_features(data)
        y = data['winner'].values
        
        # Step 3: Train-test split
        logger.info("\nSTEP 3: Train-Test Split")
        X_train, X_test, y_train, y_test = train_test_split_simple(X, y, test_size=0.2)
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Step 4: Model training
        logger.info("\nSTEP 4: Model Training")
        model = SimpleCS2Model(learning_rate=0.1, max_iter=500)
        model.feature_names = feature_names
        model.fit(X_train, y_train)
        
        # Step 5: Evaluation
        logger.info("\nSTEP 5: Model Evaluation")
        
        # Train metrics
        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)[:, 1]
        train_metrics = calculate_metrics(y_train, train_pred, train_proba)
        
        # Test metrics
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, test_pred, test_proba)
        
        logger.info("Training Results:")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Step 6: Save model
        logger.info("\nSTEP 6: Save Model")
        model_path = "models/cs2_simple_model.pkl"
        save_model(model, feature_names, norm_params, model_path)
        
        # Step 7: Validate saved model
        logger.info("\nSTEP 7: Model Validation")
        loaded_model = load_and_validate_model(model_path)
        
        # Step 8: Demo predictions
        logger.info("\nSTEP 8: Demo Predictions")
        demo_data = generate_sample_data(n_samples=3)
        demo_X, _, _ = build_features(demo_data)
        
        # Normalize using training parameters
        demo_X_norm = (demo_X - norm_params[0]) / norm_params[1]
        
        demo_pred = model.predict(demo_X_norm)
        demo_proba = model.predict_proba(demo_X_norm)
        
        logger.info("Sample Predictions:")
        for i in range(len(demo_pred)):
            team1_prob = demo_proba[i][1]
            team2_prob = demo_proba[i][0]
            predicted = "Team1" if demo_pred[i] == 1 else "Team2"
            actual = "Team1" if demo_data.iloc[i]['winner'] == 1 else "Team2"
            
            logger.info(f"  Match {i+1}: {predicted} ({team1_prob:.3f}) | Actual: {actual}")
        
        # Generate report
        logger.info("\nSTEP 9: Generate Report")
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_samples': len(data),
            'features': len(feature_names),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_path': model_path,
            'feature_names': feature_names
        }
        
        report_path = f"reports/ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {report_path}")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"📊 Test AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"💾 Model: {model_path}")
        logger.info(f"📄 Report: {report_path}")
        
        return {
            'success': True,
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['roc_auc'],
            'model_path': model_path,
            'report_path': report_path
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    results = run_complete_pipeline()
    
    if results['success']:
        print(f"\n✅ Success! Model accuracy: {results['test_accuracy']:.4f}")
        print(f"📁 Model saved to: {results['model_path']}")
    else:
        print(f"\n❌ Failed: {results['error']}")
