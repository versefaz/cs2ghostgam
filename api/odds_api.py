from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from datetime import datetime, timedelta
import os

from odds_manager import RobustOddsManager

app = FastAPI(title="Odds Service API")

# Initialize manager
odds_manager: Optional[RobustOddsManager] = None


@app.on_event("startup")
async def startup_event():
    global odds_manager
    # Initialize database and redis connections
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models.odds_history import Base

    database_url = os.getenv('DATABASE_URL') or 'postgresql://user:pass@localhost/odds_db'
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    odds_manager = RobustOddsManager(db)
    await odds_manager.initialize()


@app.get("/odds/{match_id}")
async def get_match_odds(
    match_id: str,
    force_refresh: bool = Query(False, description="Force fetch new data")
):
    """ดึงราคาล่าสุด"""
    try:
        result = await odds_manager.get_odds(match_id, force_refresh)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/odds/{match_id}/best")
async def get_best_odds(match_id: str):
    """ดึงราคาที่ดีที่สุด"""
    try:
        result = await odds_manager.get_best_odds(match_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/odds/{match_id}/history")
async def get_odds_history(
    match_id: str,
    hours: int = Query(24, description="Hours to look back")
):
    """ดึงประวัติราคา"""
    try:
        df = await odds_manager.get_odds_history(match_id, hours)
        return JSONResponse(content=df.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/odds/{match_id}/movement")
async def get_odds_movement(match_id: str):
    """ดึงการเคลื่อนไหวราคา"""
    try:
        # Get current odds first to trigger movement detection
        await odds_manager.get_odds(match_id)
        
        # Query movement data
        from models.odds_history import OddsMovement
        movements = odds_manager.db.query(OddsMovement).filter(
            OddsMovement.match_id == match_id
        ).all()
        
        result = [{
            "source": m.source,
            "odds_1_change_pct": m.odds_1_change_pct,
            "odds_2_change_pct": m.odds_2_change_pct,
            "trend": m.trend_direction,
            "volatility": m.volatility,
            "last_update": m.last_updated.isoformat() if m.last_updated else None
        } for m in movements]
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """ตรวจสอบสถานะระบบ"""
    active_sources = 0
    for source in odds_manager.fetcher.priority_sources:
        if await odds_manager._check_source_health(source):
            active_sources += 1
    
    return {
        "status": "healthy" if active_sources >= 2 else "degraded",
        "active_sources": active_sources,
        "total_sources": len(odds_manager.fetcher.priority_sources),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """ดึง metrics ของระบบ"""
    from prometheus_client import generate_latest
    return generate_latest()
