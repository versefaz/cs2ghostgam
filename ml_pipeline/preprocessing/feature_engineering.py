import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    player_features: List[str] | None = None
    team_features: List[str] | None = None
    match_features: List[str] | None = None
    temporal_features: List[str] | None = None
    interaction_features: List[str] | None = None

    def __post_init__(self):
        self.player_features = self.player_features or [
            'rating', 'kd_ratio', 'adr', 'kast', 'hs_percentage',
            'clutch_kills', 'opening_kills', 'awp_kills_per_round',
            'flash_assists', 'enemies_flashed', 'utility_damage', 'rounds_played'
        ]
        self.team_features = self.team_features or [
            'world_ranking', 'form_rating', 'map_win_rate',
            'pistol_round_win_rate', 'eco_round_win_rate',
            'force_buy_success_rate', 'avg_round_time',
            'comeback_rate', 'streak_length', 'recent_roster_changes'
        ]
        self.match_features = self.match_features or [
            'bo_type', 'tournament_tier', 'prize_pool',
            'head_to_head_score', 'map_pick_advantage',
            'days_since_last_match', 'fatigue_index',
            'team1_odds', 'team2_odds', 'team1_odds_open', 'team1_odds_close'
        ]
        self.temporal_features = self.temporal_features or [
            'hour_of_day', 'day_of_week', 'month', 'is_weekend'
        ]
        self.interaction_features = self.interaction_features or [
            'style_matchup', 'experience_diff', 'age_diff',
            'firepower_ratio', 'tactical_complexity_score'
        ]


class AdvancedFeatureEngineering:
    """Advanced feature engineering for CS2 match prediction"""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.scalers: Dict[str, object] = {}
        self.feature_importance: Dict[str, float] = {}
        self.feature_stats: Dict[str, float] = {}

    # -------------------- Player features --------------------
    def extract_player_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced player features (expects columns: player_id, team_id, rating, adr, kast, ...)."""
        if player_data.empty:
            return pd.DataFrame()

        df = player_data.sort_values(['player_id', 'date']).copy() if 'date' in player_data.columns else player_data.copy()
        grp = df.groupby('player_id')

        # Rolling window averages (30 latest rows if no date)
        if 'date' in df.columns:
            rating_30d = grp.rolling(window='30D', on='date')['rating'].mean()
        else:
            rating_30d = grp['rating'].rolling(window=30, min_periods=5).mean()
        rating_30d = rating_30d.reset_index(level=0, drop=True)

        # Trend using simple linear fit on index
        def _slope(x: pd.Series) -> float:
            if len(x) < 2:
                return 0.0
            idx = np.arange(len(x))
            try:
                m, _ = np.polyfit(idx, x.values, 1)
            except Exception:
                m = 0.0
            return float(m)

        rating_trend = grp['rating'].apply(_slope)
        rating_trend = rating_trend.reindex(df.index, method='ffill') if rating_trend.index.nlevels == 1 else rating_trend

        # Consistency and peak
        rating_std = grp['rating'].transform('std').fillna(0)
        consistency_score = 1 / (1 + rating_std)
        peak_rating = grp['rating'].transform('max').replace(0, np.nan).fillna(df['rating'])
        current_vs_peak = df['rating'] / peak_rating.replace(0, 1)

        # Role proxy scores
        rounds = df.get('rounds_played', pd.Series(1, index=df.index))
        entry_fragger_score = ((df.get('opening_kills', 0) * 2) - df.get('first_death', 0)) / rounds.replace(0, 1)
        support_score = (df.get('flash_assists', 0) * 1.5 + df.get('enemies_flashed', 0) + df.get('utility_damage', 0) / 100) / rounds.replace(0, 1)
        awper_efficiency = np.where(df.get('awp_deaths', 0).replace(0, 1) > 0,
                                    df.get('awp_kills', 0) / df.get('awp_deaths', 1), 0)

        # Clutch
        clutch_attempts = df.get('clutch_attempts', 0).replace(0, 1)
        clutch_success_rate = df.get('clutches_won', 0) / clutch_attempts
        clutch_rating = clutch_success_rate * clutch_attempts / rounds.replace(0, 1)

        # Impact
        impact_rating = df['kast'] * 0.3 + (df['adr'] / 100) * 0.3 + df['rating'] * 0.4

        # Multikill
        multi_kill_rounds = (
            df.get('2k_rounds', 0) +
            df.get('3k_rounds', 0) * 2 +
            df.get('4k_rounds', 0) * 3 +
            df.get('5k_rounds', 0) * 5
        ) / rounds.replace(0, 1)

        out = pd.DataFrame({
            'player_id': df['player_id'],
            'team_id': df.get('team_id'),
            'rating_30d': rating_30d,
            'rating_trend': rating_trend,
            'rating_std': rating_std,
            'consistency_score': consistency_score,
            'peak_rating': peak_rating,
            'current_vs_peak': current_vs_peak,
            'entry_fragger_score': entry_fragger_score,
            'support_score': support_score,
            'awper_efficiency': awper_efficiency,
            'clutch_success_rate': clutch_success_rate,
            'clutch_rating': clutch_rating,
            'impact_rating': impact_rating,
            'multi_kill_rounds': multi_kill_rounds,
        })
        return out

    # -------------------- Team features --------------------
    def extract_team_features(self, team_data: pd.DataFrame) -> pd.DataFrame:
        if team_data.empty:
            return pd.DataFrame()
        g = team_data.groupby('team_id')
        features = pd.DataFrame(index=g.size().index)
        # Example metrics; adapt as data available
        features['roster_stability'] = g['roster_changes'].apply(lambda x: 1 - (x.sum() / max(len(x), 1))) if 'roster_changes' in team_data else 1.0
        features['avg_time_together'] = g['days_together'].mean() if 'days_together' in team_data else 0
        features['strategy_diversity'] = g['strategies_used'].nunique() / g.size() if 'strategies_used' in team_data else 0
        features['eco_damage_ratio'] = (team_data.get('eco_damage_dealt', 0) / team_data.get('eco_damage_taken', 1)).groupby(team_data['team_id']).mean()
        features['force_buy_efficiency'] = (team_data.get('force_buy_rounds_won', 0) / team_data.get('force_buy_rounds', 1)).groupby(team_data['team_id']).mean()
        features['comeback_ability'] = g.apply(lambda x: (x['comebacks'].sum() / x['deficit_situations'].sum()) if ('comebacks' in x and x['deficit_situations'].sum() > 0) else 0)
        features['map_pool_depth'] = g.apply(lambda x: (x.groupby('map')['win_rate'].mean() > 0.5).sum()) if 'map' in team_data and 'win_rate' in team_data else 0
        # Fill any missing with 0
        return features.fillna(0).reset_index()

    # -------------------- Match context features --------------------
    def extract_match_context_features(self, match_data: pd.DataFrame) -> pd.DataFrame:
        if match_data.empty:
            return pd.DataFrame()
        f = pd.DataFrame(index=match_data.index)
        # Odds-based
        f['odds_implied_prob_t1'] = 1 / match_data['team1_odds']
        f['odds_implied_prob_t2'] = 1 / match_data['team2_odds']
        f['market_confidence'] = (f['odds_implied_prob_t1'] - f['odds_implied_prob_t2']).abs()
        if {'team1_odds_open', 'team1_odds_close'} <= set(match_data.columns):
            f['odds_movement'] = (match_data['team1_odds_close'] - match_data['team1_odds_open']) / match_data['team1_odds_open']
        else:
            f['odds_movement'] = 0.0
        return f

    # -------------------- Interaction features --------------------
    def create_interaction_features(self, team1_features: pd.DataFrame, team2_features: pd.DataFrame) -> pd.DataFrame:
        t1 = team1_features.add_prefix('t1_')
        t2 = team2_features.add_prefix('t2_')
        inter = pd.DataFrame(index=t1.index)
        common = set(t1.columns) & set(t2.columns)
        for c in common:
            inter[f'{c}_diff'] = t1[c] - t2[c]
            inter[f'{c}_ratio'] = t1[c] / t2[c].replace(0, 1)
        return inter.replace([np.inf, -np.inf], 0).fillna(0)

    # -------------------- Temporal features --------------------
    def engineer_temporal_features(self, match_data: pd.DataFrame) -> pd.DataFrame:
        if match_data.empty or 'match_time' not in match_data:
            return pd.DataFrame()
        ts = pd.to_datetime(match_data['match_time'])
        temporal = pd.DataFrame(index=match_data.index)
        temporal['hour'] = ts.dt.hour
        temporal['day_of_week'] = ts.dt.dayofweek
        temporal['month'] = ts.dt.month
        temporal['hour_sin'] = np.sin(2 * np.pi * temporal['hour'] / 24)
        temporal['hour_cos'] = np.cos(2 * np.pi * temporal['hour'] / 24)
        temporal['dow_sin'] = np.sin(2 * np.pi * temporal['day_of_week'] / 7)
        temporal['dow_cos'] = np.cos(2 * np.pi * temporal['day_of_week'] / 7)
        temporal['is_weekend'] = temporal['day_of_week'].isin([5, 6]).astype(int)
        return temporal

    # -------------------- Utilities --------------------
    def apply_feature_scaling(self, features: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
        if features.empty:
            return features
        if method == 'robust':
            scaler = RobustScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        num_cols = features.select_dtypes(include=[np.number]).columns
        features[num_cols] = scaler.fit_transform(features[num_cols])
        self.scalers[method] = scaler
        return features

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50, method: str = 'mutual_info') -> pd.DataFrame:
        if X.empty:
            return X
        if method != 'mutual_info':
            raise ValueError(f"Unknown selection method: {method}")
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()
        self.feature_importance = dict(zip(X.columns, selector.scores_))
        logger.info(f"Selected {len(selected_cols)} features")
        return pd.DataFrame(X_selected, columns=selected_cols)

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return features
        num_cols = features.select_dtypes(include=[np.number]).columns
        cat_cols = features.select_dtypes(exclude=[np.number]).columns
        features[num_cols] = features[num_cols].fillna(features[num_cols].median())
        for c in cat_cols:
            features[c] = features[c].fillna('unknown')
        return features

    def _remove_multicollinearity(self, features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        if features.empty:
            return features
        corr = features.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            features = features.drop(columns=to_drop, errors='ignore')
        return features

    # -------------------- End-to-end pipeline --------------------
    def create_feature_pipeline(self, match_data: pd.DataFrame, player_data: pd.DataFrame, team_data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering pipeline...")
        pf = self.extract_player_features(player_data)
        tf = self.extract_team_features(team_data)
        mf = self.extract_match_context_features(match_data)
        tempf = self.engineer_temporal_features(match_data)

        # Aggregate player features to team-level (mean by team_id)
        if not pf.empty and 'team_id' in pf.columns:
            t1 = pf.groupby('team_id').mean(numeric_only=True)
            t2 = t1.copy()
        else:
            t1 = pd.DataFrame(index=mf.index)
            t2 = pd.DataFrame(index=mf.index)

        inter = self.create_interaction_features(t1, t2) if not t1.empty and not t2.empty else pd.DataFrame(index=mf.index)

        frames = [f for f in [mf, tempf, inter] if not f.empty]
        all_features = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        all_features = self._handle_missing_values(all_features)
        all_features = self.apply_feature_scaling(all_features)
        all_features = self._remove_multicollinearity(all_features)
        logger.info(f"Feature engineering complete. Shape: {all_features.shape}")
        return all_features
