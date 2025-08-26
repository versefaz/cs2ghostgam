from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class OddsHistory(Base):
    """ตารางเก็บประวัติราคาทั้งหมด"""
    __tablename__ = 'odds_history'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(String(100), nullable=False)
    team1 = Column(String(100), nullable=False)
    team2 = Column(String(100), nullable=False)
    
    # Odds values
    odds_1 = Column(Float, nullable=False)
    odds_2 = Column(Float, nullable=False)
    odds_draw = Column(Float, nullable=True)
    
    # Metadata
    source = Column(String(50), nullable=False)
    market_type = Column(String(50), default='match_winner')
    timestamp = Column(DateTime, default=datetime.now)
    match_time = Column(DateTime, nullable=True)
    
    # Quality metrics
    confidence = Column(Float, default=1.0)
    is_validated = Column(Integer, default=0)
    
    # Raw data
    raw_data = Column(JSON, nullable=True)
    
    # Indexes for fast queries
    __table_args__ = (
        Index('idx_match_source', 'match_id', 'source'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_teams', 'team1', 'team2'),
    )


class OddsMovement(Base):
    """ตารางติดตามการเคลื่อนไหวราคา"""
    __tablename__ = 'odds_movement'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(String(100), nullable=False)
    source = Column(String(50), nullable=False)
    
    # Movement data
    odds_1_open = Column(Float)
    odds_1_current = Column(Float)
    odds_1_change = Column(Float)
    odds_1_change_pct = Column(Float)
    
    odds_2_open = Column(Float)
    odds_2_current = Column(Float)
    odds_2_change = Column(Float)
    odds_2_change_pct = Column(Float)
    
    # Timing
    first_seen = Column(DateTime)
    last_updated = Column(DateTime)
    update_count = Column(Integer, default=0)
    
    # Trend analysis
    trend_direction = Column(String(20))  # rising, falling, stable
    volatility = Column(Float)
    
    __table_args__ = (
        Index('idx_movement_match', 'match_id'),
        Index('idx_movement_update', 'last_updated'),
    )


class AggregatedOdds(Base):
    """ตารางราคาที่รวมจากหลายแหล่ง"""
    __tablename__ = 'aggregated_odds'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(String(100), nullable=False)
    
    # Aggregated values
    avg_odds_1 = Column(Float)
    avg_odds_2 = Column(Float)
    best_odds_1 = Column(Float)
    best_odds_2 = Column(Float)
    best_source_1 = Column(String(50))
    best_source_2 = Column(String(50))
    
    # Statistics
    num_sources = Column(Integer)
    sources = Column(JSON)  # List of sources
    std_dev_1 = Column(Float)  # Standard deviation
    std_dev_2 = Column(Float)
    
    # Quality
    overall_confidence = Column(Float)
    last_aggregation = Column(DateTime)
    
    # Market insights
    market_margin = Column(Float)
    implied_prob_1 = Column(Float)
    implied_prob_2 = Column(Float)
    
    __table_args__ = (
        Index('idx_agg_match', 'match_id'),
        Index('idx_agg_update', 'last_aggregation'),
    )
