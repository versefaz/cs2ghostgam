import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Load and prepare historical match data for training
    """

    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

    def load_match_data(self,
                        start_date: datetime,
                        end_date: datetime,
                        min_tier: int = 2) -> pd.DataFrame:
        """Load historical match data"""
        query = """
        SELECT 
            m.*,
            t1.name as team1_name,
            t2.name as team2_name,
            t1.world_ranking as team1_rank,
            t2.world_ranking as team2_rank
        FROM matches m
        JOIN teams t1 ON m.team1_id = t1.id
        JOIN teams t2 ON m.team2_id = t2.id
        WHERE m.date BETWEEN %s AND %s
            AND m.tournament_tier >= %s
            AND m.status = 'finished'
        ORDER BY m.date DESC
        """
        df = pd.read_sql(query, self.engine, params=(start_date, end_date, min_tier))
        logger.info(f"Loaded {len(df)} matches from {start_date} to {end_date}")
        return df

    def load_player_stats(self, match_ids: List[int]) -> pd.DataFrame:
        """Load player statistics for matches"""
        query = """
        SELECT 
            ps.*,
            p.name as player_name,
            p.team_id
        FROM player_stats ps
        JOIN players p ON ps.player_id = p.id
        WHERE ps.match_id = ANY(%s)
        """
        df = pd.read_sql(query, self.engine, params=(match_ids,))
        return df

    def load_round_data(self, match_ids: List[int]) -> pd.DataFrame:
        """Load round-by-round data"""
        query = """
        SELECT * FROM rounds
        WHERE match_id = ANY(%s)
        ORDER BY match_id, round_number
        """
        df = pd.read_sql(query, self.engine, params=(match_ids,))
        return df

    def create_training_dataset(self,
                                start_date: datetime,
                                end_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """Create complete training dataset with features and labels"""
        matches = self.load_match_data(start_date, end_date)
        match_ids = matches['match_id'].tolist()
        player_stats = self.load_player_stats(match_ids)
        round_data = self.load_round_data(match_ids)
        team_stats = self._aggregate_player_stats(player_stats)
        round_stats = self._aggregate_round_stats(round_data)
        dataset = matches
        dataset = dataset.merge(team_stats, on='match_id', how='left')
        dataset = dataset.merge(round_stats, on='match_id', how='left')
        target = (dataset['team1_score'] > dataset['team2_score']).astype(int)
        logger.info(f"Created dataset with {len(dataset)} samples")
        return dataset, target

    def _aggregate_player_stats(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player statistics by team"""
        aggregations = {
            'rating': ['mean', 'max', 'min', 'std'],
            'kills': ['sum', 'mean'],
            'deaths': ['sum', 'mean'],
            'adr': ['mean'],
            'kast': ['mean'],
            'headshot_percentage': ['mean'],
        }
        team_stats = player_stats.groupby(['match_id', 'team_id']).agg(aggregations)
        team_stats.columns = ['_'.join(col).strip() for col in team_stats.columns.values]
        team_stats = team_stats.reset_index()
        return team_stats

    def _aggregate_round_stats(self, round_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate round-level statistics"""
        round_stats = round_data.groupby('match_id').agg({
            'team1_won': 'sum',
            'team2_won': 'sum',
            'is_pistol': 'sum',
            'is_eco': 'sum',
            'bomb_planted': 'mean',
            'bomb_defused': 'mean',
        }).reset_index()
        round_stats.columns = ['match_id'] + ['rounds_' + col for col in round_stats.columns if col != 'match_id']
        return round_stats
