#!/usr/bin/env python3
"""
Default Configuration for ML Pipeline
แก้ปัญหา FeatureEngineer missing config parameter
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class FeatureConfig:
    """Default feature configuration for CS2 match prediction"""
    
    # Player-level features
    player_features: List[str] = field(default_factory=lambda: [
        'kd_ratio', 'adr', 'kpr', 'survival_rate',
        'opening_kill_ratio', 'clutch_win_rate', 'headshot_percentage',
        'first_kill_rate', 'multi_kill_rate', 'utility_damage'
    ])
    
    # Team-level features  
    team_features: List[str] = field(default_factory=lambda: [
        'win_rate', 'avg_rounds_won', 'pistol_round_win_rate',
        'eco_round_win_rate', 'force_buy_win_rate', 'anti_eco_win_rate',
        'avg_rating', 'team_kd_ratio', 'round_comeback_rate'
    ])
    
    # Map-specific features
    map_features: List[str] = field(default_factory=lambda: [
        'map_win_rate', 'ct_win_rate', 't_win_rate',
        'avg_round_duration', 'overtime_rate', 'pistol_win_impact',
        'side_preference', 'comeback_potential'
    ])
    
    # Match context features
    match_features: List[str] = field(default_factory=lambda: [
        'bo_type', 'tournament_tier', 'prize_pool',
        'head_to_head_history', 'recent_form', 'days_since_last_match',
        'travel_fatigue', 'roster_stability'
    ])
    
    # Economic features
    economic_features: List[str] = field(default_factory=lambda: [
        'save_round_efficiency', 'force_buy_success_rate',
        'eco_round_damage', 'money_management_score'
    ])
    
    # Feature engineering settings
    feature_windows: Dict[str, int] = field(default_factory=lambda: {
        'short_term': 5,    # Last 5 matches
        'medium_term': 15,  # Last 15 matches  
        'long_term': 30     # Last 30 matches
    })
    
    # Feature selection settings
    max_features: int = 50
    feature_selection_method: str = 'mutual_info'
    correlation_threshold: float = 0.95
    
    # Scaling settings
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    handle_missing: str = 'median'    # 'mean', 'median', 'drop'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for compatibility"""
        return {
            'player_features': self.player_features,
            'team_features': self.team_features,
            'map_features': self.map_features,
            'match_features': self.match_features,
            'economic_features': self.economic_features,
            'feature_windows': self.feature_windows,
            'max_features': self.max_features,
            'feature_selection_method': self.feature_selection_method,
            'correlation_threshold': self.correlation_threshold,
            'scaling_method': self.scaling_method,
            'handle_missing': self.handle_missing
        }


@dataclass 
class ModelConfig:
    """Model training configuration"""
    
    # Model types to train
    models: List[str] = field(default_factory=lambda: [
        'xgboost', 'lightgbm', 'random_forest', 'logistic_regression'
    ])
    
    # Cross-validation settings
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Training settings
    max_iterations: int = 1000
    early_stopping_rounds: int = 50
    random_state: int = 42
    
    # Hyperparameter tuning
    use_hyperopt: bool = True
    hyperopt_trials: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'models': self.models,
            'cv_folds': self.cv_folds,
            'test_size': self.test_size,
            'validation_size': self.validation_size,
            'max_iterations': self.max_iterations,
            'early_stopping_rounds': self.early_stopping_rounds,
            'random_state': self.random_state,
            'use_hyperopt': self.use_hyperopt,
            'hyperopt_trials': self.hyperopt_trials
        }


# Default instances for easy import
DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()

# Legacy compatibility
FEATURE_SETS = {
    'default': DEFAULT_FEATURE_CONFIG.to_dict(),
    'minimal': {
        'include_groups': ['basic', 'map'],
        'exclude': [],
        'max_features': 20
    },
    'comprehensive': {
        'include_groups': ['basic', 'map', 'momentum', 'economy', 'player', 'team'],
        'exclude': [],
        'max_features': 100
    }
}
