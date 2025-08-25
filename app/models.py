from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Match(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True)
    external_id = Column(String, index=True)
    team1_name = Column(String)
    team2_name = Column(String)
    team1_id = Column(Integer, ForeignKey('teams.id'), nullable=True)
    team2_id = Column(Integer, ForeignKey('teams.id'), nullable=True)
    match_time = Column(DateTime, default=datetime.utcnow)
    event_name = Column(String)
    map_name = Column(String, nullable=True)
    team1_score = Column(Integer, nullable=True)
    team2_score = Column(Integer, nullable=True)
    source = Column(String)
    status = Column(String)

class Team(Base):
    __tablename__ = 'teams'
    id = Column(Integer, primary_key=True)
    external_id = Column(String, index=True)
    name = Column(String)
    ranking = Column(Integer, nullable=True)
    region = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Player(Base):
    __tablename__ = 'players'
    id = Column(Integer, primary_key=True)
    external_id = Column(String, index=True)
    name = Column(String)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=True)
    role = Column(String, nullable=True)
    rating = Column(Float, nullable=True)
    kd_ratio = Column(Float, nullable=True)
    adr = Column(Float, nullable=True)
    kast = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PlayerPerformance(Base):
    __tablename__ = 'player_performances'
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'))
    player_name = Column(String)
    team_name = Column(String)
    kills = Column(Integer)
    deaths = Column(Integer)
    assists = Column(Integer)
    adr = Column(Float)
    kast = Column(Float)
    rating = Column(Float)

class LiveOdds(Base):
    __tablename__ = 'live_odds'
    id = Column(Integer, primary_key=True)
    match_id = Column(String)
    bookmaker = Column(String)
    team1_odds = Column(Float)
    team2_odds = Column(Float)
    scraped_at = Column(DateTime, default=datetime.utcnow)
