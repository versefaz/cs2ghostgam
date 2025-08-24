import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for CS2 match predictions
    """

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []

    def create_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature creation pipeline
        """
        logger.info("Starting feature engineering pipeline")

        # 1. Basic match features
        features = self._create_match_features(raw_data)

        # 2. Team performance features
        features = self._create_team_features(features, raw_data)

        # 3. Player performance features
        features = self._create_player_features(features, raw_data)

        # 4. Head-to-head features
        features = self._create_h2h_features(features, raw_data)

        # 5. Map-specific features
        features = self._create_map_features(features, raw_data)

        # 6. Momentum features
        features = self._create_momentum_features(features, raw_data)

        # 7. Economic features
        features = self._create_economic_features(features, raw_data)

        # 8. Temporal features
        features = self._create_temporal_features(features)

        # 9. Interaction features
        features = self._create_interaction_features(features)

        self.feature_names = features.columns.tolist()
        logger.info(f"Created {len(self.feature_names)} features")

        return features

    def _create_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic match-level features"""
        features = pd.DataFrame()

        # Match metadata
        features['match_id'] = df['match_id']
        features['tournament_tier'] = df['tournament_tier'].map({
            'S-Tier': 5, 'A-Tier': 4, 'B-Tier': 3,
            'C-Tier': 2, 'Other': 1
        })
        features['is_lan'] = df['is_lan'].astype(int)
        features['is_playoff'] = df['is_playoff'].astype(int)
        features['best_of'] = df['best_of']

        return features

    def _create_team_features(self, features: pd.DataFrame,
                              raw_data: pd.DataFrame) -> pd.DataFrame:
        """Team performance features"""

        for team in ['team1', 'team2']:
            # Recent form (last 10, 30 days)
            features[f'{team}_win_rate_10'] = self._calculate_recent_winrate(
                raw_data, team, days=10
            )
            features[f'{team}_win_rate_30'] = self._calculate_recent_winrate(
                raw_data, team, days=30
            )

            # Round statistics
            features[f'{team}_avg_rounds_won'] = raw_data[f'{team}_rounds_won_avg']
            features[f'{team}_avg_rounds_lost'] = raw_data[f'{team}_rounds_lost_avg']
            features[f'{team}_round_win_rate'] = (
                features[f'{team}_avg_rounds_won'] /
                (features[f'{team}_avg_rounds_won'] + features[f'{team}_avg_rounds_lost']).replace(0, np.nan)
            ).fillna(0.5)

            # Pistol round performance
            features[f'{team}_pistol_win_rate'] = raw_data[f'{team}_pistol_win_rate']

            # Eco round performance
            features[f'{team}_eco_win_rate'] = raw_data[f'{team}_eco_win_rate']
            features[f'{team}_anti_eco_win_rate'] = raw_data[f'{team}_anti_eco_win_rate']

            # Side performance
            features[f'{team}_t_side_win_rate'] = raw_data[f'{team}_t_win_rate']
            features[f'{team}_ct_side_win_rate'] = raw_data[f'{team}_ct_win_rate']

            # Clutch statistics
            features[f'{team}_clutch_win_rate'] = raw_data[f'{team}_clutch_win_rate']

        # Differential features
        for metric in ['win_rate_10', 'win_rate_30', 'round_win_rate',
                       'pistol_win_rate', 'eco_win_rate']:
            features[f'{metric}_diff'] = (
                features[f'team1_{metric}'] - features[f'team2_{metric}']
            )

        return features

    def _create_player_features(self, features: pd.DataFrame,
                                raw_data: pd.DataFrame) -> pd.DataFrame:
        """Player performance features"""

        for team in ['team1', 'team2']:
            features[f'{team}_avg_rating'] = raw_data[f'{team}_avg_rating']
            features[f'{team}_max_rating'] = raw_data[f'{team}_max_rating']
            features[f'{team}_min_rating'] = raw_data[f'{team}_min_rating']
            features[f'{team}_rating_std'] = raw_data[f'{team}_rating_std']

            # KD/ADR/KAST
            features[f'{team}_avg_kd'] = raw_data[f'{team}_avg_kd']
            features[f'{team}_avg_adr'] = raw_data[f'{team}_avg_adr']
            features[f'{team}_avg_kast'] = raw_data[f'{team}_avg_kast']

            # Opening duels
            features[f'{team}_opening_kill_rate'] = raw_data[f'{team}_opening_kill_rate']
            features[f'{team}_opening_death_rate'] = raw_data[f'{team}_opening_death_rate']

            # AWP performance
            features[f'{team}_awp_kills_per_round'] = raw_data[f'{team}_awp_kpr']

        # Player differential features
        features['rating_diff'] = features['team1_avg_rating'] - features['team2_avg_rating']
        features['kd_diff'] = features['team1_avg_kd'] - features['team2_avg_kd']
        features['adr_diff'] = features['team1_avg_adr'] - features['team2_avg_adr']

        return features

    def _create_h2h_features(self, features: pd.DataFrame,
                             raw_data: pd.DataFrame) -> pd.DataFrame:
        """Head-to-head historical features"""

        features['h2h_matches_played'] = raw_data['h2h_total_matches']
        features['h2h_team1_wins'] = raw_data['h2h_team1_wins']
        features['h2h_team2_wins'] = raw_data['h2h_team2_wins']

        features['h2h_team1_win_rate'] = np.where(
            features['h2h_matches_played'] > 0,
            features['h2h_team1_wins'] / features['h2h_matches_played'],
            0.5
        )

        features['h2h_recent_team1_wins'] = raw_data['h2h_recent_team1_wins']
        features['h2h_recent_matches'] = raw_data['h2h_recent_matches']

        features['h2h_map_team1_wins'] = raw_data['h2h_map_team1_wins']
        features['h2h_map_matches'] = raw_data['h2h_map_matches']

        return features

    def _create_map_features(self, features: pd.DataFrame,
                             raw_data: pd.DataFrame) -> pd.DataFrame:
        """Map-specific features"""

        for team in ['team1', 'team2']:
            for map_name in ['mirage', 'inferno', 'dust2', 'nuke',
                             'overpass', 'ancient', 'vertigo', 'anubis']:
                features[f'{team}_{map_name}_win_rate'] = raw_data[f'{team}_{map_name}_wr']
                features[f'{team}_{map_name}_matches'] = raw_data[f'{team}_{map_name}_matches']

        current_map = raw_data['current_map'].iloc[0] if 'current_map' in raw_data.columns else None
        if current_map:
            features['team1_current_map_wr'] = features[f'team1_{current_map}_win_rate']
            features['team2_current_map_wr'] = features[f'team2_{current_map}_win_rate']
            features['current_map_wr_diff'] = (
                features['team1_current_map_wr'] - features['team2_current_map_wr']
            )

        features['team1_map_pick_wr'] = raw_data['team1_map_pick_win_rate']
        features['team2_map_pick_wr'] = raw_data['team2_map_pick_win_rate']

        return features

    def _create_momentum_features(self, features: pd.DataFrame,
                                  raw_data: pd.DataFrame) -> pd.DataFrame:
        """Momentum and streak features"""

        for team in ['team1', 'team2']:
            features[f'{team}_current_streak'] = raw_data[f'{team}_current_streak']
            features[f'{team}_is_winning_streak'] = (
                raw_data[f'{team}_current_streak'] > 0
            ).astype(int)

            features[f'{team}_form_trajectory'] = self._calculate_form_trajectory(
                raw_data, team
            )

            features[f'{team}_days_since_last'] = raw_data[f'{team}_days_since_last_match']
            features[f'{team}_matches_last_week'] = raw_data[f'{team}_matches_last_7d']

        features['streak_diff'] = features['team1_current_streak'] - features['team2_current_streak']
        features['form_trajectory_diff'] = (
            features['team1_form_trajectory'] - features['team2_form_trajectory']
        )

        return features

    def _create_economic_features(self, features: pd.DataFrame,
                                  raw_data: pd.DataFrame) -> pd.DataFrame:
        """Economic and equipment value features"""

        if 'current_money_team1' in raw_data.columns:
            features['team1_current_money'] = raw_data['current_money_team1']
            features['team2_current_money'] = raw_data['current_money_team2']
            features['money_diff'] = features['team1_current_money'] - features['team2_current_money']

            features['team1_equipment_value'] = raw_data['equipment_value_team1']
            features['team2_equipment_value'] = raw_data['equipment_value_team2']
            features['equipment_diff'] = (
                features['team1_equipment_value'] - features['team2_equipment_value']
            )

            features['team1_loss_bonus'] = raw_data['loss_bonus_team1']
            features['team2_loss_bonus'] = raw_data['loss_bonus_team2']

        features['team1_avg_economy_rating'] = raw_data.get('team1_economy_rating', 0)
        features['team2_avg_economy_rating'] = raw_data.get('team2_economy_rating', 0)

        return features

    def _create_temporal_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""

        if 'match_time' in features.columns:
            features['hour_of_day'] = pd.to_datetime(features['match_time']).dt.hour
            features['day_of_week'] = pd.to_datetime(features['match_time']).dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

        features['days_since_update'] = 30

        return features

    def _create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Feature interactions and polynomial features"""

        features['rating_kd_interaction'] = features['rating_diff'] * features['kd_diff']
        features['form_streak_interaction'] = (
            features['team1_form_trajectory'] * features['team1_current_streak']
        )

        if 'is_lan' in features.columns:
            features['lan_rating_boost'] = features['is_lan'] * features['rating_diff']

        features['playoff_pressure'] = (
            features['is_playoff'] * features['tournament_tier']
        )

        features['rating_diff_squared'] = features['rating_diff'] ** 2
        features['win_rate_diff_squared'] = features['win_rate_10_diff'] ** 2

        return features

    def _calculate_recent_winrate(self, data: pd.DataFrame,
                                  team: str, days: int) -> pd.Series:
        """Calculate team's win rate in last N days (placeholder)"""
        return data[f'{team}_winrate_{days}d'].fillna(0.5)

    def _calculate_form_trajectory(self, data: pd.DataFrame,
                                   team: str) -> pd.Series:
        """Calculate if team is improving or declining"""
        recent = data[f'{team}_rating_7d']
        average = data[f'{team}_rating_30d']
        return (recent - average).replace(0, np.nan) / average.replace(0, np.nan)

    def normalize_features(self, features: pd.DataFrame,
                           fit: bool = True) -> pd.DataFrame:
        """Normalize numerical features"""
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['match_id']]
        if fit:
            features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        else:
            features[numerical_cols] = self.scaler.transform(features[numerical_cols])
        return features

    def select_features(self, features: pd.DataFrame,
                        target: pd.Series,
                        k: int = 50) -> pd.DataFrame:
        """Select top k features based on statistical tests"""
        selector = SelectKBest(f_classif, k=k)
        selected_features = selector.fit_transform(features, target)
        selected_indices = selector.get_support(indices=True)
        selected_columns = features.columns[selected_indices]
        logger.info(f"Selected {len(selected_columns)} features")
        return pd.DataFrame(selected_features, columns=selected_columns)

    def save_feature_metadata(self, filepath: str):
        """Save feature names and preprocessing objects"""
        import pickle
        metadata = {
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'config': self.config,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved feature metadata to {filepath}")
