from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseWriter:
    def __init__(self, connection_string: str = None):
        """Initialize database writer with connection pooling"""
        self.connection_string = connection_string or \
            'postgresql://postgres:password@localhost:5432/cs2_betting'

        # Create engine with connection pooling
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set True for debugging
        )

        self.Session = sessionmaker(bind=self.engine)
        self.metadata = None

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                _ = conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection successful")
        except SQLAlchemyError as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    def write_match_data(self, data: Dict[str, Any]) -> bool:
        """Write match data to database"""
        with self.get_session() as session:
            try:
                # Prepare match data
                match_data = {
                    'external_match_id': data.get('match_id'),
                    'date': datetime.fromisoformat(data.get('date', datetime.utcnow().isoformat())),
                    'team1_name': data.get('team1'),
                    'team2_name': data.get('team2'),
                    'tournament_name': data.get('tournament'),
                    'best_of': data.get('best_of', 3),
                    'status': data.get('status', 'scheduled'),
                    'score_team1': data.get('score_team1'),
                    'score_team2': data.get('score_team2'),
                    'odds_team1': data.get('odds_team1'),
                    'odds_team2': data.get('odds_team2'),
                    'maps': json.dumps(data.get('maps', [])),
                    'raw_data': json.dumps(data),
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }

                # Upsert match
                query = text("""
                    INSERT INTO matches (
                        external_match_id, date, team1_name, team2_name,
                        tournament_name, best_of, status,
                        score_team1, score_team2,
                        odds_team1, odds_team2,
                        maps, raw_data, created_at, updated_at
                    ) VALUES (
                        :external_match_id, :date, :team1_name, :team2_name,
                        :tournament_name, :best_of, :status,
                        :score_team1, :score_team2,
                        :odds_team1, :odds_team2,
                        :maps, :raw_data, :created_at, :updated_at
                    )
                    ON CONFLICT (external_match_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        score_team1 = EXCLUDED.score_team1,
                        score_team2 = EXCLUDED.score_team2,
                        odds_team1 = EXCLUDED.odds_team1,
                        odds_team2 = EXCLUDED.odds_team2,
                        maps = EXCLUDED.maps,
                        raw_data = EXCLUDED.raw_data,
                        updated_at = EXCLUDED.updated_at
                    RETURNING id
                """)

                result = session.execute(query, match_data)
                match_id = result.scalar()

                logger.info(f"✅ Match {data.get('match_id')} saved (ID: {match_id})")
                return True

            except SQLAlchemyError as e:
                logger.error(f"❌ Failed to write match data: {e}")
                return False

    def write_player_stats(self, match_id: str, stats: List[Dict]) -> bool:
        """Write player statistics for a match"""
        with self.get_session() as session:
            try:
                for player_stat in stats:
                    stat_data = {
                        'match_id': match_id,
                        'player_name': player_stat.get('name'),
                        'team_name': player_stat.get('team'),
                        'kills': player_stat.get('kills', 0),
                        'deaths': player_stat.get('deaths', 0),
                        'assists': player_stat.get('assists', 0),
                        'adr': player_stat.get('adr', 0.0),
                        'kast': player_stat.get('kast', 0.0),
                        'rating': player_stat.get('rating', 0.0),
                        'headshot_percentage': player_stat.get('hs_percent', 0.0),
                        'created_at': datetime.utcnow()
                    }

                    query = text("""
                        INSERT INTO player_stats (
                            match_id, player_name, team_name,
                            kills, deaths, assists, adr, kast, rating,
                            headshot_percentage, created_at
                        ) VALUES (
                            :match_id, :player_name, :team_name,
                            :kills, :deaths, :assists, :adr, :kast, :rating,
                            :headshot_percentage, :created_at
                        )
                        ON CONFLICT (match_id, player_name) DO UPDATE SET
                            kills = EXCLUDED.kills,
                            deaths = EXCLUDED.deaths,
                            assists = EXCLUDED.assists,
                            adr = EXCLUDED.adr,
                            kast = EXCLUDED.kast,
                            rating = EXCLUDED.rating,
                            headshot_percentage = EXCLUDED.headshot_percentage
                    """)

                    session.execute(query, stat_data)

                logger.info(f"✅ Player stats saved for match {match_id}")
                return True

            except SQLAlchemyError as e:
                logger.error(f"❌ Failed to write player stats: {e}")
                return False

    def write_odds_history(self, data: Dict[str, Any]) -> bool:
        """Track odds movement over time"""
        with self.get_session() as session:
            try:
                odds_data = {
                    'match_id': data.get('match_id'),
                    'bookmaker': data.get('bookmaker', 'pinnacle'),
                    'odds_team1': data.get('odds_team1'),
                    'odds_team2': data.get('odds_team2'),
                    'odds_draw': data.get('odds_draw'),
                    'timestamp': datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat()))
                }

                query = text("""
                    INSERT INTO odds_history (
                        match_id, bookmaker, odds_team1, odds_team2, 
                        odds_draw, timestamp
                    ) VALUES (
                        :match_id, :bookmaker, :odds_team1, :odds_team2,
                        :odds_draw, :timestamp
                    )
                """)

                session.execute(query, odds_data)
                logger.info(f"✅ Odds history saved for match {data.get('match_id')}")
                return True

            except SQLAlchemyError as e:
                logger.error(f"❌ Failed to write odds history: {e}")
                return False

    def write_batch(self, table_name: str, data_list: List[Dict]) -> bool:
        """Batch write data to any table"""
        if not data_list:
            return True

        try:
            # Convert to DataFrame for efficient batch insert
            df = pd.DataFrame(data_list)
            df['created_at'] = datetime.utcnow()
            df['updated_at'] = datetime.utcnow()

            # Batch insert
            df.to_sql(
                table_name,
                con=self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )

            logger.info(f"✅ Batch inserted {len(data_list)} records to {table_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Batch write failed: {e}")
            return False

    def get_recent_matches(self, limit: int = 10) -> List[Dict]:
        """Get recent matches for monitoring"""
        with self.get_session() as session:
            query = text("""
                SELECT 
                    external_match_id,
                    date,
                    team1_name,
                    team2_name,
                    score_team1,
                    score_team2,
                    status,
                    odds_team1,
                    odds_team2
                FROM matches
                ORDER BY date DESC
                LIMIT :limit
            """)

            result = session.execute(query, {'limit': limit})
            return [dict(row) for row in result]

    def health_check(self) -> Dict[str, Any]:
        """Database health check"""
        try:
            with self.get_session() as session:
                # Check connection
                _ = session.execute(text("SELECT 1"))

                # Get table stats
                stats = session.execute(text("""
                    SELECT 
                        'matches' as table_name,
                        COUNT(*) as row_count
                    FROM matches
                    UNION ALL
                    SELECT 
                        'player_stats' as table_name,
                        COUNT(*) as row_count
                    FROM player_stats
                """))

                table_stats = {row['table_name']: row['row_count'] for row in stats}

                pool = self.engine.pool
                return {
                    'status': 'healthy',
                    'tables': table_stats,
                    'connection_pool': {
                        'size': getattr(pool, 'size', lambda: None)(),
                        'checked_in': getattr(pool, 'checkedin', lambda: None)(),
                        'overflow': getattr(pool, 'overflow', lambda: None)()
                    }
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
