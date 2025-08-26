import asyncio
from odds_manager import RobustOddsManager

# You need to pass a real SQLAlchemy session here
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.odds_history import Base

engine = create_engine('postgresql://odds_user:secure_password@localhost/odds_db')
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
db_session = SessionLocal()


async def main():
    # Initialize manager
    manager = RobustOddsManager(db_session)
    await manager.initialize()
    
    # Get odds for a match
    match_id = "hltv_12345"
    odds = await manager.get_odds(match_id)
    
    print(f"Match: {match_id}")
    print(f"Sources: {odds['metadata']['sources_count']}")
    # Aggregated structure: Dict keyed by team matchup; pick first
    agg_values = next(iter(odds['aggregated'].values())) if odds.get('aggregated') else {}
    print(f"Best odds Team 1: {agg_values.get('best_odds_1')}")
    print(f"Best odds Team 2: {agg_values.get('best_odds_2')}")
    if odds.get('movement') and odds['movement'].get('summary'):
        print(f"Movement: {odds['movement']['summary'].get('market_sentiment')}")
    
    # Get historical data
    history = await manager.get_odds_history(match_id, hours_back=48)
    print(f"History records: {len(history)}")

if __name__ == "__main__":
    asyncio.run(main())
