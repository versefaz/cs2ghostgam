"""
Model configuration and parameters
"""

MODEL_CONFIG = {
    'baseline': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
        },
        'logistic_regression': {
            'penalty': 'l2',
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
        },
    },
    'advanced': {
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        },
        'lightgbm': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'random_state': 42,
            'verbosity': -1,
        },
        'neural_network': {
            'hidden_layers': [128, 64, 32],
            'activation': 'relu',
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping': True,
            'patience': 10,
        },
    },
    'ensemble': {
        'voting_type': 'soft',
        'weights': [0.3, 0.4, 0.3],  # RF, XGB, LGB
        'meta_learner': 'logistic_regression',
    },
    'optimization': {
        'n_trials': 100,
        'timeout': 3600,  # 1 hour
        'n_jobs': -1,
        'sampler': 'TPESampler',
        'pruner': 'MedianPruner',
    },
    'training': {
        'test_size': 0.2,
        'validation_size': 0.2,
        'cv_folds': 5,
        'random_state': 42,
        'stratify': True,
    },
    'thresholds': {
        'min_confidence': 0.55,
        'max_bet_size': 0.1,  # 10% of bankroll
        'kelly_cap': 0.25,
    },
}

# Feature groups (placeholder groups; refine as data columns stabilize)
FEATURE_GROUPS = {
    'basic': [
        'team1_win_rate_10', 'team2_win_rate_10',
        'team1_avg_rating', 'team2_avg_rating',
        'rating_diff', 'best_of', 'tournament_tier', 'is_lan', 'is_playoff',
    ],
    'map': [
        'team1_current_map_wr', 'team2_current_map_wr', 'current_map_wr_diff',
        'team1_map_pick_wr', 'team2_map_pick_wr',
    ],
    'momentum': [
        'team1_current_streak', 'team2_current_streak', 'streak_diff',
        'team1_form_trajectory', 'team2_form_trajectory', 'form_trajectory_diff',
    ],
    'economy': [
        'team1_current_money', 'team2_current_money', 'money_diff',
        'team1_equipment_value', 'team2_equipment_value', 'equipment_diff',
    ],
}
