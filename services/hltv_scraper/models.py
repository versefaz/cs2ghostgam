"""Database models for HLTV Scraper"""
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(String(50), unique=True, nullable=False, index=True)
    team1_name = Column(String(100), nullable=False)
    team2_name = Column(String(100), nullable=False)
    team1_id = Column(Integer, ForeignKey("teams.id"))
    team2_id = Column(Integer, ForeignKey("teams.id"))
    event_name = Column(String(200))
    format = Column(String(10))
    scheduled_time = Column(DateTime(timezone=True))
    status = Column(String(20), default="upcoming")
    team1_score = Column(Integer)
    team2_score = Column(Integer)

    # Metadata
    scraped_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    team1 = relationship("Team", foreign_keys=[team1_id], back_populates="home_matches")
    team2 = relationship("Team", foreign_keys=[team2_id], back_populates="away_matches")
    performances = relationship("PlayerPerformance", back_populates="match")
    odds = relationship("LiveOdds", back_populates="match")
    predictions = relationship("Prediction", back_populates="match")

    # Indexes
    __table_args__ = (
        Index('idx_match_scheduled', 'scheduled_time'),
        Index('idx_match_status', 'status'),
        Index('idx_match_teams', 'team1_id', 'team2_id'),
    )

class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False, index=True)
    country = Column(String(50))
    world_ranking = Column(Integer)
    rating_2_0 = Column(Float)
    total_players = Column(Integer, default=5)

    # Stats
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float)
    average_age = Column(Float)
    map_stats = Column(JSON)
    recent_form = Column(JSON)

    # Metadata
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scraped = Column(DateTime(timezone=True))

    # Relationships
    home_matches = relationship("Match", foreign_keys="Match.team1_id", back_populates="team1")
    away_matches = relationship("Match", foreign_keys="Match.team2_id", back_populates="team2")
    players = relationship("Player", back_populates="team")
    performances = relationship("PlayerPerformance", back_populates="team")

    # Indexes
    __table_args__ = (
        Index('idx_team_ranking', 'world_ranking'),
        Index('idx_team_rating', 'rating_2_0'),
    )

class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String(50), unique=True, nullable=False, index=True)
    nickname = Column(String(50), nullable=False, index=True)
    real_name = Column(String(100))
    country = Column(String(50))
    age = Column(Integer)
    current_team_id = Column(Integer, ForeignKey("teams.id"))

    # Stats
    rating_2_0 = Column(Float)
    kd_ratio = Column(Float)
    kpr = Column(Float)  # Kills per round
    adr = Column(Float)  # Average damage per round
    kast = Column(Float)  # Percentage of rounds with kill, assist, survival, or traded death
    hs_percentage = Column(Float)  # Headshot percentage
    maps_played = Column(Integer, default=0)
    rounds_played = Column(Integer, default=0)

    # Form stats
    rating_last_3_months = Column(Float)
    form_rating = Column(Float)
    peak_rating = Column(Float)

    # Metadata
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scraped = Column(DateTime(timezone=True))

    # Relationships
    team = relationship("Team", back_populates="players")
    performances = relationship("PlayerPerformance", back_populates="player")

    # Indexes
    __table_args__ = (
        Index('idx_player_rating', 'rating_2_0'),
        Index('idx_player_team', 'current_team_id'),
    )

class PlayerPerformance(Base):
    __tablename__ = "player_performances"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # Map info
    map_name = Column(String(50))
    map_number = Column(Integer)

    # Performance stats
    kills = Column(Integer)
    deaths = Column(Integer)
    assists = Column(Integer)
    kd_diff = Column(Integer)
    adr = Column(Float)
    kast = Column(Float)
    rating_2_0 = Column(Float)

    # Round stats
    first_kills = Column(Integer)
    first_deaths = Column(Integer)
    clutches_won = Column(Integer)

    # Weapon stats
    hs_count = Column(Integer)
    hs_percentage = Column(Float)
    awp_kills = Column(Integer)

    # Economy
    equipment_value = Column(Integer)
    money_saved = Column(Integer)

    # Metadata
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    match = relationship("Match", back_populates="performances")
    player = relationship("Player", back_populates="performances")
    team = relationship("Team", back_populates="performances")

    # Indexes
    __table_args__ = (
        Index('idx_performance_match_player', 'match_id', 'player_id'),
        Index('idx_performance_rating', 'rating_2_0'),
        Index('idx_performance_timestamp', 'timestamp'),
    )

class LiveOdds(Base):
    __tablename__ = "live_odds"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker = Column(String(50), nullable=False)

    # Odds
    team1_odds = Column(Float)
    team2_odds = Column(Float)
    draw_odds = Column(Float)

    # Map odds
    over_maps = Column(Float)
    under_maps = Column(Float)

    # Special markets
    first_map_winner_team1 = Column(Float)
    first_map_winner_team2 = Column(Float)
    correct_score_odds = Column(JSON)

    # Metadata
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    is_live = Column(Boolean, default=False)

    # Relationships
    match = relationship("Match", back_populates="odds")

    # Indexes
    __table_args__ = (
        Index('idx_odds_match_time', 'match_id', 'timestamp'),
        Index('idx_odds_bookmaker', 'bookmaker'),
    )

class ScrapeJob(Base):
    __tablename__ = "scrape_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(50), unique=True, nullable=False, index=True)
    job_type = Column(String(50), nullable=False)  # matches, teams, players, live
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed

    # Job details
    target_url = Column(String(500))
    parameters = Column(JSON)
    priority = Column(Integer, default=5)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Results
    items_scraped = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)
    error_message = Column(String(1000))
    result_data = Column(JSON)

    # Timing
    scheduled_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)

    # Metadata
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))

    # Indexes
    __table_args__ = (
        Index('idx_job_status', 'status'),
        Index('idx_job_type', 'job_type'),
        Index('idx_job_scheduled', 'scheduled_at'),
        Index('idx_job_priority', 'priority'),
    )

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)

    # Model info
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)

    # Predictions
    predicted_winner = Column(String(100))
    team1_win_probability = Column(Float)
    team2_win_probability = Column(Float)

    # Score predictions
    predicted_score_team1 = Column(Integer)
    predicted_score_team2 = Column(Integer)
    predicted_maps = Column(Integer)

    # Map predictions
    map_predictions = Column(JSON)  # Detailed predictions per map

    # Confidence and accuracy
    confidence_score = Column(Float)
    prediction_metadata = Column(JSON)

    # Result tracking
    actual_winner = Column(String(100))
    was_correct = Column(Boolean)

    # Metadata
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    match = relationship("Match", back_populates="predictions")

    # Indexes
    __table_args__ = (
        Index('idx_prediction_match', 'match_id'),
        Index('idx_prediction_model', 'model_name', 'model_version'),
        Index('idx_prediction_confidence', 'confidence_score'),
        Index('idx_prediction_created', 'created_at'),
    )

class ProxyStatus(Base):
    __tablename__ = "proxy_status"

    id = Column(Integer, primary_key=True, index=True)
    proxy_host = Column(String(50), nullable=False)
    proxy_port = Column(Integer, nullable=False)
    proxy_type = Column(String(20))  # http, socks5, residential

    # Authentication
    username = Column(String(100))
    password = Column(String(100))

    # Status
    is_active = Column(Boolean, default=True)
    is_banned = Column(Boolean, default=False)
    health_score = Column(Float, default=100.0)

    # Performance metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    average_response_time = Column(Float)

    # Ban tracking
    ban_count = Column(Integer, default=0)
    last_ban_time = Column(DateTime(timezone=True))
    ban_duration_hours = Column(Integer)

    # Metadata
    last_used = Column(DateTime(timezone=True))
    last_tested = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_proxy_active', 'is_active'),
        Index('idx_proxy_health', 'health_score'),
        Index('idx_proxy_host_port', 'proxy_host', 'proxy_port', unique=True),
    )

class ScrapingMetrics(Base):
    __tablename__ = "scraping_metrics"

    id = Column(Integer, primary_key=True, index=True)

    # Metric identification
    metric_type = Column(String(50), nullable=False)  # response_time, success_rate, etc.
    target_type = Column(String(50), nullable=False)  # matches, teams, players, etc.

    # Values
    value = Column(Float, nullable=False)
    count = Column(Integer, default=1)

    # Aggregations
    min_value = Column(Float)
    max_value = Column(Float)
    avg_value = Column(Float)
    p50_value = Column(Float)
    p95_value = Column(Float)
    p99_value = Column(Float)

    # Context
    proxy_used = Column(String(100))
    error_type = Column(String(100))
    http_status = Column(Integer)

    # Metadata
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_metrics_timestamp', 'timestamp'),
        Index('idx_metrics_type', 'metric_type', 'target_type'),
    )
