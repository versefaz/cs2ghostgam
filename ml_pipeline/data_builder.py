"""
CS2 Data Builder - Comprehensive Feature Engineering for ML Models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Feature categories to include
    include_player_features: bool = True
    include_team_features: bool = True
    include_map_features: bool = True
    include_context_features: bool = True
    include_advanced_features: bool = True
    
    # Feature scaling and preprocessing
    scale_features: bool = True
    handle_missing: str = 'median'  # 'median', 'mean', 'zero'
    
    # Feature selection
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.001

class CS2DataBuilder:
    """
    Comprehensive data builder for CS2 betting models
    Generates 100+ features from match data including player stats, team performance,
    map analysis, tournament context, and advanced engineered features
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_names = []
        self.feature_importance = {}
        self.scaler = None
        
        # Initialize scaler if needed
        if self.config.scale_features:
            try:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            except ImportError:
                logger.warning("scikit-learn not available, features will not be scaled")
                self.scaler = None
        
        logger.info("CS2DataBuilder initialized")
    
    def build_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive feature set from raw match data
        
        Args:
            raw_data: DataFrame with match information
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Building features for {len(raw_data)} matches")
        
        features = pd.DataFrame(index=raw_data.index)
        
        # Player-level features (40+ features)
        if self.config.include_player_features:
            player_features = self._build_player_features(raw_data)
            features = pd.concat([features, player_features], axis=1)
        
        # Team-level features (30+ features)
        if self.config.include_team_features:
            team_features = self._build_team_features(raw_data)
            features = pd.concat([features, team_features], axis=1)
        
        # Map performance features (25+ features)
        if self.config.include_map_features:
            map_features = self._build_map_features(raw_data)
            features = pd.concat([features, map_features], axis=1)
        
        # Context features (20+ features)
        if self.config.include_context_features:
            context_features = self._build_context_features(raw_data)
            features = pd.concat([features, context_features], axis=1)
        
        # Advanced engineered features (15+ features)
        if self.config.include_advanced_features:
            advanced_features = self._build_advanced_features(raw_data, features)
            features = pd.concat([features, advanced_features], axis=1)
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Generated {len(self.feature_names)} features")
        return features
    
    def _build_player_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build player-specific features (40+ features)"""
        features = pd.DataFrame(index=data.index)
        
        # Individual player stats for both teams
        for team_prefix in ['team1', 'team2']:
            team_stats = []
            
            for player_idx in range(1, 6):  # 5 players per team
                prefix = f"{team_prefix}_p{player_idx}"
                
                # Core stats
                kills = data.get(f"{prefix}_kills", 0)
                deaths = data.get(f"{prefix}_deaths", 1).replace(0, 1)  # Avoid division by zero
                assists = data.get(f"{prefix}_assists", 0)
                adr = data.get(f"{prefix}_adr", 70)
                rating = data.get(f"{prefix}_rating", 1.0)
                
                # Store individual stats
                features[f"{prefix}_kills"] = kills
                features[f"{prefix}_deaths"] = deaths
                features[f"{prefix}_assists"] = assists
                features[f"{prefix}_adr"] = adr
                features[f"{prefix}_rating"] = rating
                features[f"{prefix}_kd_ratio"] = kills / deaths
                
                # Role-specific stats
                features[f"{prefix}_entry_frags"] = data.get(f"{prefix}_entry_frags", 0)
                features[f"{prefix}_clutches_won"] = data.get(f"{prefix}_clutches_won", 0)
                features[f"{prefix}_awp_kills"] = data.get(f"{prefix}_awp_kills", 0)
                features[f"{prefix}_headshot_pct"] = data.get(f"{prefix}_headshot_pct", 0.45)
                
                # Collect for team aggregations
                team_stats.append({
                    'kills': kills,
                    'deaths': deaths,
                    'assists': assists,
                    'adr': adr,
                    'rating': rating
                })
            
            # Team aggregations
            team_df = pd.DataFrame(team_stats)
            features[f"{team_prefix}_total_kills"] = team_df['kills'].sum()
            features[f"{team_prefix}_total_deaths"] = team_df['deaths'].sum()
            features[f"{team_prefix}_avg_adr"] = team_df['adr'].mean()
            features[f"{team_prefix}_avg_rating"] = team_df['rating'].mean()
            features[f"{team_prefix}_team_kd"] = team_df['kills'].sum() / team_df['deaths'].sum()
            features[f"{team_prefix}_rating_std"] = team_df['rating'].std()
        
        return features
    
    def _build_team_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build team-level features (30+ features)"""
        features = pd.DataFrame(index=data.index)
        
        # Win rates and form
        features['team1_win_rate_30d'] = data.get('team1_win_rate_30d', 0.5)
        features['team2_win_rate_30d'] = data.get('team2_win_rate_30d', 0.5)
        features['team1_win_rate_90d'] = data.get('team1_win_rate_90d', 0.5)
        features['team2_win_rate_90d'] = data.get('team2_win_rate_90d', 0.5)
        features['team1_recent_form'] = data.get('team1_recent_form', 0.5)
        features['team2_recent_form'] = data.get('team2_recent_form', 0.5)
        
        # Round performance
        features['team1_avg_rounds_won'] = data.get('team1_avg_rounds_won', 13)
        features['team2_avg_rounds_won'] = data.get('team2_avg_rounds_won', 13)
        features['team1_avg_rounds_lost'] = data.get('team1_avg_rounds_lost', 13)
        features['team2_avg_rounds_lost'] = data.get('team2_avg_rounds_lost', 13)
        
        # Specialized round types
        features['team1_pistol_win_rate'] = data.get('team1_pistol_win_rate', 0.5)
        features['team2_pistol_win_rate'] = data.get('team2_pistol_win_rate', 0.5)
        features['team1_eco_win_rate'] = data.get('team1_eco_win_rate', 0.2)
        features['team2_eco_win_rate'] = data.get('team2_eco_win_rate', 0.2)
        features['team1_force_win_rate'] = data.get('team1_force_win_rate', 0.3)
        features['team2_force_win_rate'] = data.get('team2_force_win_rate', 0.3)
        
        # Side performance
        features['team1_ct_win_rate'] = data.get('team1_ct_win_rate', 0.5)
        features['team2_ct_win_rate'] = data.get('team2_ct_win_rate', 0.5)
        features['team1_t_win_rate'] = data.get('team1_t_win_rate', 0.5)
        features['team2_t_win_rate'] = data.get('team2_t_win_rate', 0.5)
        
        # Team composition and stability
        features['team1_roster_changes_30d'] = data.get('team1_roster_changes_30d', 0)
        features['team2_roster_changes_30d'] = data.get('team2_roster_changes_30d', 0)
        features['team1_avg_age'] = data.get('team1_avg_age', 23)
        features['team2_avg_age'] = data.get('team2_avg_age', 23)
        features['team1_avg_experience'] = data.get('team1_avg_experience', 3)
        features['team2_avg_experience'] = data.get('team2_avg_experience', 3)
        
        # Performance differentials
        features['win_rate_diff_30d'] = features['team1_win_rate_30d'] - features['team2_win_rate_30d']
        features['win_rate_diff_90d'] = features['team1_win_rate_90d'] - features['team2_win_rate_90d']
        features['form_diff'] = features['team1_recent_form'] - features['team2_recent_form']
        features['rounds_won_diff'] = features['team1_avg_rounds_won'] - features['team2_avg_rounds_won']
        features['pistol_diff'] = features['team1_pistol_win_rate'] - features['team2_pistol_win_rate']
        features['ct_performance_diff'] = features['team1_ct_win_rate'] - features['team2_ct_win_rate']
        features['t_performance_diff'] = features['team1_t_win_rate'] - features['team2_t_win_rate']
        
        return features
    
    def _build_map_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build map-specific features (25+ features)"""
        features = pd.DataFrame(index=data.index)
        
        # Current map being played
        current_map = data.get('current_map', 'dust2')
        features['current_map_team1_win_rate'] = data.get('current_map_team1_win_rate', 0.5)
        features['current_map_team2_win_rate'] = data.get('current_map_team2_win_rate', 0.5)
        features['current_map_diff'] = features['current_map_team1_win_rate'] - features['current_map_team2_win_rate']
        
        # Map pool analysis for major maps
        maps = ['dust2', 'mirage', 'inferno', 'nuke', 'overpass', 'ancient', 'vertigo']
        
        for map_name in maps:
            # Team 1 map performance
            features[f'team1_{map_name}_played'] = data.get(f'team1_{map_name}_played', 0)
            features[f'team1_{map_name}_win_rate'] = data.get(f'team1_{map_name}_win_rate', 0.5)
            features[f'team1_{map_name}_avg_score'] = data.get(f'team1_{map_name}_avg_score', 13)
            
            # Team 2 map performance
            features[f'team2_{map_name}_played'] = data.get(f'team2_{map_name}_played', 0)
            features[f'team2_{map_name}_win_rate'] = data.get(f'team2_{map_name}_win_rate', 0.5)
            features[f'team2_{map_name}_avg_score'] = data.get(f'team2_{map_name}_avg_score', 13)
        
        # Map pool strength indicators
        features['team1_map_pool_depth'] = sum([
            (data.get(f'team1_{map_name}_played', 0) > 5).astype(int) for map_name in maps
        ])
        features['team2_map_pool_depth'] = sum([
            (data.get(f'team2_{map_name}_played', 0) > 5).astype(int) for map_name in maps
        ])
        
        # Map experience differential
        features['map_experience_diff'] = (
            data.get('team1_current_map_played', 0) - data.get('team2_current_map_played', 0)
        )
        
        return features
    
    def _build_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build match context features (20+ features)"""
        features = pd.DataFrame(index=data.index)
        
        # Recent form and streaks
        features['team1_last_5_wins'] = data.get('team1_last_5_wins', 2.5)
        features['team2_last_5_wins'] = data.get('team2_last_5_wins', 2.5)
        features['team1_current_streak'] = data.get('team1_current_streak', 0)
        features['team2_current_streak'] = data.get('team2_current_streak', 0)
        
        # Head-to-head history
        features['h2h_team1_wins'] = data.get('h2h_team1_wins', 0)
        features['h2h_team2_wins'] = data.get('h2h_team2_wins', 0)
        features['h2h_total_matches'] = features['h2h_team1_wins'] + features['h2h_team2_wins']
        features['h2h_team1_win_rate'] = np.where(
            features['h2h_total_matches'] > 0,
            features['h2h_team1_wins'] / features['h2h_total_matches'],
            0.5
        )
        
        # Tournament and match context
        features['tournament_tier'] = data.get('tournament_tier', 2)  # 1-5 scale
        features['prize_pool'] = data.get('prize_pool', 100000)
        features['is_lan'] = data.get('is_lan', 0)
        features['is_elimination'] = data.get('is_elimination', 0)
        features['is_final'] = data.get('is_final', 0)
        features['best_of'] = data.get('best_of', 1)
        
        # Timing factors
        features['team1_days_since_last'] = data.get('team1_days_since_last', 7)
        features['team2_days_since_last'] = data.get('team2_days_since_last', 7)
        features['match_hour'] = data.get('match_hour', 18)
        features['match_day_of_week'] = data.get('match_day_of_week', 3)
        
        # Rankings and seeding
        features['team1_rank'] = data.get('team1_rank', 15)
        features['team2_rank'] = data.get('team2_rank', 15)
        features['rank_difference'] = features['team1_rank'] - features['team2_rank']
        features['seeding_diff'] = data.get('team1_seed', 8) - data.get('team2_seed', 8)
        
        return features
    
    def _build_advanced_features(self, data: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
        """Build advanced engineered features (15+ features)"""
        features = pd.DataFrame(index=data.index)
        
        # Momentum indicators
        team1_momentum = data.get('team1_recent_form', 0.5) * (data.get('team1_current_streak', 0) / 5)
        team2_momentum = data.get('team2_recent_form', 0.5) * (data.get('team2_current_streak', 0) / 5)
        features['team1_momentum'] = team1_momentum
        features['team2_momentum'] = team2_momentum
        features['momentum_diff'] = team1_momentum - team2_momentum
        
        # Pressure and stakes indicators
        tournament_pressure = data.get('tournament_tier', 2) * data.get('is_elimination', 0)
        features['team1_pressure'] = tournament_pressure * (1 / (data.get('team1_rank', 15) + 1))
        features['team2_pressure'] = tournament_pressure * (1 / (data.get('team2_rank', 15) + 1))
        
        # Experience in high-stakes matches
        features['team1_big_match_exp'] = (
            data.get('team1_finals_played', 0) + data.get('team1_semifinals_played', 0)
        )
        features['team2_big_match_exp'] = (
            data.get('team2_finals_played', 0) + data.get('team2_semifinals_played', 0)
        )
        features['experience_diff'] = features['team1_big_match_exp'] - features['team2_big_match_exp']
        
        # Fatigue indicators
        features['team1_fatigue'] = (
            data.get('team1_matches_last_7d', 0) / (data.get('team1_days_since_last', 7) + 1)
        )
        features['team2_fatigue'] = (
            data.get('team2_matches_last_7d', 0) / (data.get('team2_days_since_last', 7) + 1)
        )
        features['fatigue_diff'] = features['team1_fatigue'] - features['team2_fatigue']
        
        # Composite performance scores
        features['team1_composite_score'] = (
            data.get('team1_win_rate_30d', 0.5) * 0.3 +
            base_features.get('team1_avg_rating', 1.0) * 0.3 +
            (1 / (data.get('team1_rank', 15) + 1)) * 0.2 +
            data.get('team1_recent_form', 0.5) * 0.2
        )
        features['team2_composite_score'] = (
            data.get('team2_win_rate_30d', 0.5) * 0.3 +
            base_features.get('team2_avg_rating', 1.0) * 0.3 +
            (1 / (data.get('team2_rank', 15) + 1)) * 0.2 +
            data.get('team2_recent_form', 0.5) * 0.2
        )
        features['composite_diff'] = features['team1_composite_score'] - features['team2_composite_score']
        
        # Odds-based features (if available)
        if 'team1_odds' in data.columns and 'team2_odds' in data.columns:
            features['team1_implied_prob'] = 1 / data['team1_odds']
            features['team2_implied_prob'] = 1 / data['team2_odds']
            features['odds_diff'] = features['team1_implied_prob'] - features['team2_implied_prob']
            features['market_confidence'] = abs(features['odds_diff'])
        
        return features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        if self.config.handle_missing == 'median':
            return features.fillna(features.median())
        elif self.config.handle_missing == 'mean':
            return features.fillna(features.mean())
        elif self.config.handle_missing == 'zero':
            return features.fillna(0)
        else:
            return features
    
    def create_labels(self, data: pd.DataFrame, label_column: str = 'winner') -> np.ndarray:
        """
        Create binary labels for training
        
        Args:
            data: DataFrame with match outcomes
            label_column: Column containing match results
            
        Returns:
            Binary labels (1 for team1 win, 0 for team2 win)
        """
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        
        # Convert to binary labels
        labels = (data[label_column] == 'team1').astype(int).values
        
        team1_win_rate = np.mean(labels)
        logger.info(f"Created labels: {len(labels)} samples, {team1_win_rate:.2%} team1 wins")
        
        return labels
    
    def prepare_data(self, raw_data: pd.DataFrame, label_column: str = 'winner') -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline
        
        Args:
            raw_data: Raw match data
            label_column: Column with match outcomes
            
        Returns:
            Tuple of (features_array, labels_array)
        """
        logger.info("Starting data preparation pipeline")
        
        # Build features
        features_df = self.build_features(raw_data)
        
        # Create labels
        labels = self.create_labels(raw_data, label_column)
        
        # Scale features if configured
        if self.scaler is not None and self.config.scale_features:
            features_array = self.scaler.fit_transform(features_df)
            logger.info("Features scaled using StandardScaler")
        else:
            features_array = features_df.values
        
        logger.info(f"Data preparation complete: {features_array.shape[0]} samples, {features_array.shape[1]} features")
        
        return features_array, labels
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores if available"""
        return self.feature_importance.copy()
