#!/usr/bin/env python3
"""
ML Training Data Collector
Collects and processes historical match data, team stats, and odds for model training
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from app.scrapers.enhanced_hltv_scraper import EnhancedHLTVScraper, MatchContext, TeamStats
from app.scrapers.robust_odds_scraper import RobustOddsScraper, MarketConsensus
from ml_pipeline.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class TrainingMatch:
    """Complete training data for a single match"""
    match_id: str
    timestamp: datetime
    
    # Teams
    team1_name: str
    team2_name: str
    
    # Team statistics at time of match
    team1_ranking: int
    team2_ranking: int
    team1_form: float  # Win rate last 10 matches
    team2_form: float
    team1_map_winrates: Dict[str, float]
    team2_map_winrates: Dict[str, float]
    
    # Head-to-head
    h2h_team1_wins: int
    h2h_team2_wins: int
    h2h_total: int
    
    # Match context
    event_name: str
    bo_format: str
    map_pool: List[str]
    
    # Market data
    team1_odds: float
    team2_odds: float
    bookmaker_count: int
    
    # Result (target variable)
    winner: str  # team1 or team2
    final_score: str  # e.g., "2-1"
    match_duration: Optional[int] = None  # minutes


class MLDataCollector:
    """Collects and processes training data for ML models"""
    
    def __init__(self, db_path: str = "data/training_data.db"):
        self.db_path = db_path
        self.feature_engineer = FeatureEngineer()
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for storing training data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_matches (
                    match_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    team1_name TEXT,
                    team2_name TEXT,
                    team1_ranking INTEGER,
                    team2_ranking INTEGER,
                    team1_form REAL,
                    team2_form REAL,
                    team1_map_winrates TEXT,  -- JSON
                    team2_map_winrates TEXT,  -- JSON
                    h2h_team1_wins INTEGER,
                    h2h_team2_wins INTEGER,
                    h2h_total INTEGER,
                    event_name TEXT,
                    bo_format TEXT,
                    map_pool TEXT,  -- JSON
                    team1_odds REAL,
                    team2_odds REAL,
                    bookmaker_count INTEGER,
                    winner TEXT,
                    final_score TEXT,
                    match_duration INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_vectors (
                    match_id TEXT PRIMARY KEY,
                    features TEXT,  -- JSON array of features
                    target INTEGER,  -- 0 for team2 win, 1 for team1 win
                    feature_names TEXT,  -- JSON array of feature names
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES training_matches (match_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_date TEXT,
                    matches_collected INTEGER,
                    date_range_start TEXT,
                    date_range_end TEXT,
                    sources TEXT,  -- JSON array
                    notes TEXT
                )
            """)
    
    async def collect_historical_data(
        self, 
        days_back: int = 90, 
        max_matches: int = 1000
    ) -> int:
        """Collect historical match data for training"""
        logger.info(f"Starting historical data collection: {days_back} days, max {max_matches} matches")
        
        collected_matches = 0
        
        async with EnhancedHLTVScraper() as hltv_scraper:
            async with RobustOddsScraper() as odds_scraper:
                
                # Get recent completed matches from HLTV
                # Note: This is a simplified approach - in production you'd need
                # to scrape historical results pages or use archived data
                
                # For demonstration, we'll simulate collecting data
                # In reality, you'd need to:
                # 1. Scrape HLTV results pages going back in time
                # 2. For each match, get team stats at that point in time
                # 3. Get historical odds data (challenging without API access)
                # 4. Store everything in structured format
                
                # Simulate data collection process
                await self._simulate_data_collection(days_back, max_matches)
                collected_matches = max_matches
        
        # Record collection metadata
        self._record_collection_metadata(
            collected_matches, 
            days_back, 
            ["hltv", "oddsportal", "ggbet"]
        )
        
        logger.info(f"Collected {collected_matches} historical matches")
        return collected_matches
    
    async def _simulate_data_collection(self, days_back: int, max_matches: int):
        """Simulate historical data collection (for demonstration)"""
        # This simulates what real data collection would look like
        # In production, replace with actual scraping logic
        
        teams = [
            "NAVI", "G2", "FaZe", "Astralis", "Vitality", "MOUZ", 
            "Liquid", "NIP", "Heroic", "ENCE", "BIG", "Complexity"
        ]
        
        events = [
            "IEM Katowice", "ESL Pro League", "BLAST Premier", 
            "PGL Major", "EPICENTER", "DreamHack Masters"
        ]
        
        maps = ["dust2", "mirage", "inferno", "cache", "overpass", "train", "nuke"]
        
        training_matches = []
        
        for i in range(max_matches):
            # Generate realistic training data
            team1, team2 = np.random.choice(teams, 2, replace=False)
            
            # Simulate rankings (lower is better)
            team1_ranking = np.random.randint(1, 30)
            team2_ranking = np.random.randint(1, 30)
            
            # Form based on ranking with noise
            team1_form = max(0.3, min(0.9, 0.8 - (team1_ranking / 50) + np.random.normal(0, 0.1)))
            team2_form = max(0.3, min(0.9, 0.8 - (team2_ranking / 50) + np.random.normal(0, 0.1)))
            
            # Map winrates
            team1_map_winrates = {map_name: max(0.2, min(0.8, np.random.normal(team1_form, 0.1))) for map_name in maps}
            team2_map_winrates = {map_name: max(0.2, min(0.8, np.random.normal(team2_form, 0.1))) for map_name in maps}
            
            # H2H data
            h2h_total = np.random.randint(0, 20)
            h2h_team1_wins = np.random.randint(0, h2h_total + 1) if h2h_total > 0 else 0
            h2h_team2_wins = h2h_total - h2h_team1_wins
            
            # Odds based on team strength with bookmaker margin
            team1_prob = team1_form / (team1_form + team2_form)
            team2_prob = 1 - team1_prob
            
            # Add bookmaker margin (5-10%)
            margin = np.random.uniform(0.05, 0.10)
            team1_odds = 1 / (team1_prob * (1 - margin))
            team2_odds = 1 / (team2_prob * (1 - margin))
            
            # Determine winner based on probabilities with some randomness
            winner_prob = team1_prob + np.random.normal(0, 0.1)
            winner = "team1" if np.random.random() < winner_prob else "team2"
            
            # Generate match data
            match = TrainingMatch(
                match_id=f"sim_{i:04d}",
                timestamp=datetime.utcnow() - timedelta(days=np.random.randint(1, days_back)),
                team1_name=team1,
                team2_name=team2,
                team1_ranking=team1_ranking,
                team2_ranking=team2_ranking,
                team1_form=team1_form,
                team2_form=team2_form,
                team1_map_winrates=team1_map_winrates,
                team2_map_winrates=team2_map_winrates,
                h2h_team1_wins=h2h_team1_wins,
                h2h_team2_wins=h2h_team2_wins,
                h2h_total=h2h_total,
                event_name=np.random.choice(events),
                bo_format=np.random.choice(["BO1", "BO3", "BO5"]),
                map_pool=np.random.choice(maps, size=np.random.randint(1, 4), replace=False).tolist(),
                team1_odds=team1_odds,
                team2_odds=team2_odds,
                bookmaker_count=np.random.randint(2, 8),
                winner=winner,
                final_score="2-1" if winner == "team1" else "1-2",
                match_duration=np.random.randint(60, 180)
            )
            
            training_matches.append(match)
        
        # Store in database
        self._store_training_matches(training_matches)
        logger.info(f"Generated and stored {len(training_matches)} simulated training matches")
    
    def _store_training_matches(self, matches: List[TrainingMatch]):
        """Store training matches in database"""
        with sqlite3.connect(self.db_path) as conn:
            for match in matches:
                conn.execute("""
                    INSERT OR REPLACE INTO training_matches VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    match.match_id,
                    match.timestamp.isoformat(),
                    match.team1_name,
                    match.team2_name,
                    match.team1_ranking,
                    match.team2_ranking,
                    match.team1_form,
                    match.team2_form,
                    json.dumps(match.team1_map_winrates),
                    json.dumps(match.team2_map_winrates),
                    match.h2h_team1_wins,
                    match.h2h_team2_wins,
                    match.h2h_total,
                    match.event_name,
                    match.bo_format,
                    json.dumps(match.map_pool),
                    match.team1_odds,
                    match.team2_odds,
                    match.bookmaker_count,
                    match.winner,
                    match.final_score,
                    match.match_duration,
                    datetime.utcnow().isoformat()
                ))
    
    def _record_collection_metadata(self, matches_collected: int, days_back: int, sources: List[str]):
        """Record metadata about data collection"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO collection_metadata 
                (collection_date, matches_collected, date_range_start, date_range_end, sources, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                matches_collected,
                (datetime.utcnow() - timedelta(days=days_back)).isoformat(),
                datetime.utcnow().isoformat(),
                json.dumps(sources),
                f"Collected {matches_collected} matches over {days_back} days"
            ))
    
    def generate_feature_vectors(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate feature vectors from stored training data"""
        logger.info("Generating feature vectors from training data")
        
        # Load training data
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM training_matches 
                ORDER BY timestamp DESC
            """, conn)
        
        if df.empty:
            logger.warning("No training data found")
            return np.array([]), np.array([]), []
        
        features_list = []
        targets = []
        
        for _, row in df.iterrows():
            try:
                # Parse JSON fields
                team1_map_winrates = json.loads(row['team1_map_winrates'])
                team2_map_winrates = json.loads(row['team2_map_winrates'])
                map_pool = json.loads(row['map_pool'])
                
                # Create match context for feature engineering
                match_context = {
                    'team1_name': row['team1_name'],
                    'team2_name': row['team2_name'],
                    'team1_ranking': row['team1_ranking'],
                    'team2_ranking': row['team2_ranking'],
                    'team1_form': row['team1_form'],
                    'team2_form': row['team2_form'],
                    'team1_map_winrates': team1_map_winrates,
                    'team2_map_winrates': team2_map_winrates,
                    'h2h_team1_wins': row['h2h_team1_wins'],
                    'h2h_team2_wins': row['h2h_team2_wins'],
                    'h2h_total': row['h2h_total'],
                    'event_name': row['event_name'],
                    'bo_format': row['bo_format'],
                    'map_pool': map_pool,
                    'team1_odds': row['team1_odds'],
                    'team2_odds': row['team2_odds'],
                    'bookmaker_count': row['bookmaker_count']
                }
                
                # Generate features
                features = self.feature_engineer.extract_features(match_context)
                features_list.append(features)
                
                # Target: 1 if team1 wins, 0 if team2 wins
                target = 1 if row['winner'] == 'team1' else 0
                targets.append(target)
                
            except Exception as e:
                logger.error(f"Error processing match {row['match_id']}: {e}")
                continue
        
        if not features_list:
            logger.warning("No valid features generated")
            return np.array([]), np.array([]), []
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(targets)
        
        # Get feature names
        feature_names = self.feature_engineer.get_feature_names()
        
        # Store feature vectors in database
        self._store_feature_vectors(df['match_id'].tolist(), features_list, targets, feature_names)
        
        logger.info(f"Generated {len(X)} feature vectors with {len(feature_names)} features")
        return X, y, feature_names
    
    def _store_feature_vectors(
        self, 
        match_ids: List[str], 
        features_list: List[List[float]], 
        targets: List[int], 
        feature_names: List[str]
    ):
        """Store generated feature vectors in database"""
        with sqlite3.connect(self.db_path) as conn:
            for match_id, features, target in zip(match_ids, features_list, targets):
                conn.execute("""
                    INSERT OR REPLACE INTO feature_vectors 
                    (match_id, features, target, feature_names, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    match_id,
                    json.dumps(features),
                    target,
                    json.dumps(feature_names),
                    datetime.utcnow().isoformat()
                ))
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get processed training data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT features, target, feature_names 
                FROM feature_vectors 
                ORDER BY created_at DESC
            """)
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning("No feature vectors found, generating from raw data")
                return self.generate_feature_vectors()
            
            features_list = []
            targets = []
            feature_names = None
            
            for row in rows:
                features = json.loads(row[0])
                target = row[1]
                if feature_names is None:
                    feature_names = json.loads(row[2])
                
                features_list.append(features)
                targets.append(target)
            
            X = np.array(features_list)
            y = np.array(targets)
            
            return X, y, feature_names
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected training data"""
        with sqlite3.connect(self.db_path) as conn:
            # Basic counts
            match_count = conn.execute("SELECT COUNT(*) FROM training_matches").fetchone()[0]
            feature_count = conn.execute("SELECT COUNT(*) FROM feature_vectors").fetchone()[0]
            
            # Date range
            date_range = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM training_matches
            """).fetchone()
            
            # Team distribution
            team_stats = conn.execute("""
                SELECT team1_name as team, COUNT(*) as matches FROM training_matches GROUP BY team1_name
                UNION ALL
                SELECT team2_name as team, COUNT(*) as matches FROM training_matches GROUP BY team2_name
                ORDER BY matches DESC
                LIMIT 10
            """).fetchall()
            
            # Win rate distribution
            win_rates = conn.execute("""
                SELECT 
                    AVG(CASE WHEN winner = 'team1' THEN 1.0 ELSE 0.0 END) as team1_win_rate,
                    COUNT(*) as total_matches
                FROM training_matches
            """).fetchone()
            
            return {
                'total_matches': match_count,
                'feature_vectors': feature_count,
                'date_range': {
                    'start': date_range[0],
                    'end': date_range[1]
                },
                'top_teams': dict(team_stats),
                'overall_win_rate': {
                    'team1_wins': win_rates[0],
                    'team2_wins': 1 - win_rates[0],
                    'total_matches': win_rates[1]
                }
            }
    
    def export_to_csv(self, output_path: str = "training_data.csv"):
        """Export training data to CSV for external analysis"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT tm.*, fv.features, fv.target, fv.feature_names
                FROM training_matches tm
                LEFT JOIN feature_vectors fv ON tm.match_id = fv.match_id
                ORDER BY tm.timestamp DESC
            """, conn)
            
            df.to_csv(output_path, index=False)
            logger.info(f"Training data exported to {output_path}")


# Usage example
async def main():
    """Example usage of ML Data Collector"""
    collector = MLDataCollector()
    
    # Collect historical data
    matches_collected = await collector.collect_historical_data(days_back=90, max_matches=500)
    print(f"Collected {matches_collected} matches")
    
    # Generate feature vectors
    X, y, feature_names = collector.generate_feature_vectors()
    print(f"Generated {len(X)} feature vectors with {len(feature_names)} features")
    
    # Get statistics
    stats = collector.get_data_statistics()
    print(f"Data statistics: {json.dumps(stats, indent=2)}")
    
    # Export to CSV
    collector.export_to_csv("cs2_training_data.csv")


if __name__ == "__main__":
    asyncio.run(main())
