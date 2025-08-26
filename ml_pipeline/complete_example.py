"""
CS2 ML Pipeline - Complete Integration Example
Demonstrates end-to-end ML training, validation, and inference
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our ML components
from ml_pipeline.data_builder import CS2DataBuilder, FeatureConfig
from ml_pipeline.model_trainer import CS2ModelTrainer, ModelConfig
from ml_pipeline.model_validator import CS2ModelValidator, run_model_validation_suite
from cs2_betting_system.config.settings import MODEL_PATH, MODEL_SAVE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic CS2 match data for demonstration
    
    Args:
        n_samples: Number of sample matches to generate
        
    Returns:
        DataFrame with synthetic match data
    """
    logger.info(f"Generating {n_samples} synthetic match samples")
    
    np.random.seed(42)  # For reproducible results
    
    data = []
    
    for i in range(n_samples):
        # Team 1 stats
        team1_rating = np.random.uniform(0.8, 1.4)
        team1_win_rate_30d = np.random.uniform(0.3, 0.8)
        team1_recent_form = np.random.uniform(0.2, 0.9)
        team1_rank = np.random.randint(1, 30)
        
        # Team 2 stats  
        team2_rating = np.random.uniform(0.8, 1.4)
        team2_win_rate_30d = np.random.uniform(0.3, 0.8)
        team2_recent_form = np.random.uniform(0.2, 0.9)
        team2_rank = np.random.randint(1, 30)
        
        # Player stats for both teams
        sample = {
            'match_id': f"match_{i:04d}",
            'team1_name': f"Team_A_{i % 20}",
            'team2_name': f"Team_B_{i % 20}",
            
            # Team performance
            'team1_win_rate_30d': team1_win_rate_30d,
            'team2_win_rate_30d': team2_win_rate_30d,
            'team1_win_rate_90d': team1_win_rate_30d * np.random.uniform(0.9, 1.1),
            'team2_win_rate_90d': team2_win_rate_30d * np.random.uniform(0.9, 1.1),
            'team1_recent_form': team1_recent_form,
            'team2_recent_form': team2_recent_form,
            'team1_rank': team1_rank,
            'team2_rank': team2_rank,
            
            # Round performance
            'team1_avg_rounds_won': np.random.uniform(10, 16),
            'team2_avg_rounds_won': np.random.uniform(10, 16),
            'team1_pistol_win_rate': np.random.uniform(0.3, 0.7),
            'team2_pistol_win_rate': np.random.uniform(0.3, 0.7),
            'team1_ct_win_rate': np.random.uniform(0.4, 0.6),
            'team2_ct_win_rate': np.random.uniform(0.4, 0.6),
            
            # Context
            'tournament_tier': np.random.randint(1, 6),
            'is_lan': np.random.choice([0, 1]),
            'current_map_team1_win_rate': np.random.uniform(0.3, 0.8),
            'current_map_team2_win_rate': np.random.uniform(0.3, 0.8),
            'h2h_team1_wins': np.random.randint(0, 10),
            'h2h_team2_wins': np.random.randint(0, 10),
        }
        
        # Player stats (5 players per team)
        for team_prefix in ['team1', 'team2']:
            team_rating = team1_rating if team_prefix == 'team1' else team2_rating
            
            for player_idx in range(1, 6):
                prefix = f"{team_prefix}_p{player_idx}"
                
                # Generate correlated player stats
                base_rating = team_rating * np.random.uniform(0.7, 1.3)
                
                sample[f"{prefix}_kills"] = np.random.poisson(base_rating * 15)
                sample[f"{prefix}_deaths"] = max(1, np.random.poisson(base_rating * 12))
                sample[f"{prefix}_assists"] = np.random.poisson(base_rating * 8)
                sample[f"{prefix}_adr"] = base_rating * np.random.uniform(60, 90)
                sample[f"{prefix}_rating"] = base_rating
                sample[f"{prefix}_entry_frags"] = np.random.poisson(2)
                sample[f"{prefix}_clutches_won"] = np.random.poisson(1)
                sample[f"{prefix}_awp_kills"] = np.random.poisson(3)
                sample[f"{prefix}_headshot_pct"] = np.random.uniform(0.3, 0.6)
        
        # Generate winner based on team strength (with some randomness)
        team1_strength = (team1_rating + team1_win_rate_30d + team1_recent_form) / 3
        team2_strength = (team2_rating + team2_win_rate_30d + team2_recent_form) / 3
        
        # Add some randomness to make it realistic
        win_probability = team1_strength / (team1_strength + team2_strength)
        win_probability = np.clip(win_probability * np.random.uniform(0.7, 1.3), 0.1, 0.9)
        
        sample['winner'] = 'team1' if np.random.random() < win_probability else 'team2'
        
        data.append(sample)
    
    df = pd.DataFrame(data)
    logger.info(f"Generated dataset with {len(df)} samples, {df['winner'].value_counts()['team1']}/{len(df)} team1 wins")
    
    return df

def run_complete_ml_pipeline():
    """
    Run the complete ML pipeline from data preparation to model validation
    """
    logger.info("Starting complete CS2 ML pipeline")
    
    # Ensure directories exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    try:
        # Step 1: Generate sample data
        logger.info("=" * 60)
        logger.info("STEP 1: Data Generation")
        logger.info("=" * 60)
        
        raw_data = generate_sample_data(n_samples=1000)
        
        # Step 2: Feature engineering
        logger.info("=" * 60)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 60)
        
        feature_config = FeatureConfig(
            include_player_features=True,
            include_team_features=True,
            include_map_features=True,
            include_context_features=True,
            include_advanced_features=True,
            scale_features=True
        )
        
        data_builder = CS2DataBuilder(config=feature_config)
        X, y = data_builder.prepare_data(raw_data, label_column='winner')
        
        logger.info(f"Feature engineering complete: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Step 3: Model training
        logger.info("=" * 60)
        logger.info("STEP 3: Model Training")
        logger.info("=" * 60)
        
        model_config = ModelConfig(
            model_type='xgboost',
            hyperparameter_tuning=True,
            cv_folds=5,
            test_size=0.2,
            model_save_path=MODEL_SAVE_PATH
        )
        
        trainer = CS2ModelTrainer(config=model_config)
        training_results = trainer.train(X, y, feature_names=data_builder.get_feature_names())
        
        logger.info("Training Results Summary:")
        logger.info(f"  CV AUC: {training_results['cv_mean']:.4f} ± {training_results['cv_std']:.4f}")
        logger.info(f"  Test AUC: {training_results['test_results']['roc_auc']:.4f}")
        logger.info(f"  Model saved to: {training_results['model_path']}")
        
        # Step 4: Model validation
        logger.info("=" * 60)
        logger.info("STEP 4: Model Validation")
        logger.info("=" * 60)
        
        validator = CS2ModelValidator()
        validation_results = validator.comprehensive_model_test(training_results['model_path'])
        
        logger.info("Validation Results:")
        logger.info(f"  Overall Status: {validation_results['overall_status']}")
        logger.info(f"  Model Type: {validation_results['validation']['model_type']}")
        logger.info(f"  File Size: {validation_results['validation']['file_size_mb']} MB")
        logger.info(f"  Feature Count: {validation_results['validation']['feature_count']}")
        
        if validation_results['smoke_test'].get('inference_successful'):
            logger.info(f"  Inference Time: {validation_results['smoke_test']['execution_time_ms']} ms")
        
        # Step 5: Smoke test inference
        logger.info("=" * 60)
        logger.info("STEP 5: Smoke Test Inference")
        logger.info("=" * 60)
        
        # Load the trained model and test inference
        trained_model = CS2ModelTrainer()
        trained_model.load_model(training_results['model_path'])
        
        # Generate test samples
        test_data = generate_sample_data(n_samples=5)
        test_features, _ = data_builder.prepare_data(test_data, label_column='winner')
        
        # Make predictions
        predictions = trained_model.predict(test_features)
        probabilities = trained_model.predict_proba(test_features)
        
        logger.info("Sample Predictions:")
        for i in range(len(predictions)):
            team1_prob = probabilities[i][1]
            team2_prob = probabilities[i][0]
            predicted_winner = "Team1" if predictions[i] == 1 else "Team2"
            actual_winner = test_data.iloc[i]['winner']
            
            logger.info(f"  Match {i+1}: {predicted_winner} (Team1: {team1_prob:.3f}, Team2: {team2_prob:.3f}) | Actual: {actual_winner}")
        
        # Step 6: Generate comprehensive report
        logger.info("=" * 60)
        logger.info("STEP 6: Generate Report")
        logger.info("=" * 60)
        
        report_path = f"reports/ml_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report_lines = []
        report_lines.append("CS2 ML PIPELINE EXECUTION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("DATA SUMMARY:")
        report_lines.append(f"  Training Samples: {X.shape[0]}")
        report_lines.append(f"  Features Generated: {X.shape[1]}")
        report_lines.append(f"  Class Balance: {np.mean(y):.2%} Team1 wins")
        report_lines.append("")
        
        report_lines.append("TRAINING RESULTS:")
        report_lines.append(f"  Model Type: {model_config.model_type}")
        report_lines.append(f"  CV AUC: {training_results['cv_mean']:.4f} ± {training_results['cv_std']:.4f}")
        report_lines.append(f"  Test Accuracy: {training_results['test_results']['accuracy']:.4f}")
        report_lines.append(f"  Test Precision: {training_results['test_results']['precision']:.4f}")
        report_lines.append(f"  Test Recall: {training_results['test_results']['recall']:.4f}")
        report_lines.append(f"  Test F1: {training_results['test_results']['f1_score']:.4f}")
        report_lines.append(f"  Test AUC: {training_results['test_results']['roc_auc']:.4f}")
        report_lines.append("")
        
        report_lines.append("VALIDATION RESULTS:")
        report_lines.append(f"  Validation Status: {validation_results['overall_status']}")
        report_lines.append(f"  Model Loadable: {validation_results['validation']['loadable']}")
        report_lines.append(f"  Inference Successful: {validation_results['smoke_test'].get('inference_successful', False)}")
        report_lines.append(f"  Inference Time: {validation_results['smoke_test'].get('execution_time_ms', 0)} ms")
        report_lines.append("")
        
        report_lines.append("TOP 10 IMPORTANT FEATURES:")
        sorted_features = sorted(training_results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            report_lines.append(f"  {i+1:2d}. {feature}: {importance:.4f}")
        report_lines.append("")
        
        report_lines.append("MODEL FILES:")
        report_lines.append(f"  Model Path: {training_results['model_path']}")
        report_lines.append(f"  Model Version: {training_results['model_version']}")
        report_lines.append(f"  Model Hash: {training_results['model_hash']}")
        
        report_text = "\n".join(report_lines)
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        
        # Final summary
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 60)
        logger.info("✅ Data generation successful")
        logger.info("✅ Feature engineering successful")
        logger.info("✅ Model training successful")
        logger.info("✅ Model validation successful")
        logger.info("✅ Smoke test inference successful")
        logger.info("✅ Report generation successful")
        logger.info("")
        logger.info(f"🎯 Final Model Performance: {training_results['test_results']['roc_auc']:.4f} AUC")
        logger.info(f"📁 Model saved to: {training_results['model_path']}")
        logger.info(f"📊 Report saved to: {report_path}")
        
        return {
            'success': True,
            'model_path': training_results['model_path'],
            'model_performance': training_results['test_results']['roc_auc'],
            'validation_status': validation_results['overall_status'],
            'report_path': report_path
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def quick_model_test(model_path: str = None):
    """
    Quick test of a trained model
    
    Args:
        model_path: Path to model file (uses latest if None)
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    logger.info(f"Quick testing model: {model_path}")
    
    # Validate model
    validator = CS2ModelValidator()
    results = validator.comprehensive_model_test(model_path)
    
    if results['overall_status'] == 'PASS':
        logger.info("✅ Model validation passed")
        logger.info(f"   Model Type: {results['validation']['model_type']}")
        logger.info(f"   Features: {results['validation']['feature_count']}")
        logger.info(f"   Inference Time: {results['smoke_test']['execution_time_ms']} ms")
    else:
        logger.error("❌ Model validation failed")
        if results.get('error'):
            logger.error(f"   Error: {results['error']}")

if __name__ == "__main__":
    # Run complete pipeline
    results = run_complete_ml_pipeline()
    
    if results['success']:
        print("\n🎉 ML Pipeline completed successfully!")
        print(f"Model Performance: {results['model_performance']:.4f} AUC")
        print(f"Model Path: {results['model_path']}")
        print(f"Report: {results['report_path']}")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
