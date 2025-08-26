from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager

from shared.redis_schema import RedisSchemaManager, KeyType

logger = logging.getLogger(__name__)


class PredictionQuery(BaseModel):
    team: Optional[str] = None
    min_confidence: Optional[float] = None
    min_expected_value: Optional[float] = None
    limit: Optional[int] = 50


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]
    redis_health: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting API Gateway...")
    yield
    # Shutdown
    logger.info("Shutting down API Gateway...")


app = FastAPI(
    title="CS2 Betting System API",
    description="API for CS2 betting predictions, signals, and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
schema_manager = RedisSchemaManager(redis_client)


def get_redis_manager():
    return schema_manager


@app.get("/")
async def root():
    return {"message": "CS2 Betting System API", "version": "1.0.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        # Redis health
        redis_health = schema_manager.health_check()
        
        # Service health checks
        services = {
            "api_gateway": {"status": "healthy", "timestamp": datetime.now().isoformat()},
            "redis": {"status": redis_health.get("status", "unknown")},
            "prediction_tracker": await _check_service_health("prediction_tracker"),
            "scraper": await _check_service_health("scraper"),
        }
        
        overall_status = "healthy" if all(
            s.get("status") == "healthy" for s in services.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            services=services,
            redis_health=redis_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _check_service_health(service_name: str) -> Dict[str, Any]:
    """Check health of a specific service"""
    try:
        health_data = schema_manager.get_with_json(KeyType.HEALTH, 'service', service_name=service_name)
        
        if not health_data:
            return {"status": "unknown", "message": "No health data available"}
        
        # Check if health data is recent (within last 2 minutes)
        last_update = datetime.fromisoformat(health_data.get('timestamp', ''))
        if (datetime.now() - last_update).total_seconds() > 120:
            return {"status": "stale", "message": "Health data is stale", "last_update": health_data.get('timestamp')}
        
        return {
            "status": health_data.get('status', 'unknown'),
            "timestamp": health_data.get('timestamp'),
            "metrics": health_data.get('metrics', {})
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/predictions/recent")
async def get_recent_predictions(
    limit: int = Query(50, ge=1, le=200),
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get recent predictions"""
    try:
        # Get from recent predictions list
        recent_data = redis_client.lrange('predictions:recent', 0, limit - 1)
        
        predictions = []
        for data in recent_data:
            try:
                prediction = json.loads(data)
                predictions.append(prediction)
            except json.JSONDecodeError:
                continue
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictions/search")
async def search_predictions(
    query: PredictionQuery,
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Search predictions with filters"""
    try:
        query_dict = query.dict(exclude_none=True)
        results = schema.search_predictions(query_dict)
        
        # Apply limit
        limit = query_dict.get('limit', 50)
        results = results[:limit]
        
        return {
            "predictions": results,
            "count": len(results),
            "query": query_dict,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{match_id}")
async def get_prediction(
    match_id: str,
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get prediction for specific match"""
    try:
        prediction = schema.get_with_json(KeyType.PREDICTION, 'current', match_id=match_id)
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Also get prediction history
        history = schema.get_with_json(KeyType.PREDICTION, 'history', match_id=match_id) or []
        
        return {
            "prediction": prediction,
            "history": history,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction for {match_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/queue")
async def get_signal_queue_status(
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get signal queue statistics"""
    try:
        queue_stats = schema.get_queue_stats()
        
        return {
            "queue_stats": queue_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/active")
async def get_active_signals(
    limit: int = Query(50, ge=1, le=200),
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get active signals"""
    try:
        # Get active signal keys
        signal_keys = schema.list_keys(KeyType.SIGNAL, 'active')
        
        signals = []
        for key in signal_keys[:limit]:
            signal_data = redis_client.get(key)
            if signal_data:
                try:
                    signals.append(json.loads(signal_data))
                except json.JSONDecodeError:
                    continue
        
        return {
            "signals": signals,
            "count": len(signals),
            "total_active": len(signal_keys),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/live")
async def get_live_matches(
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get live matches"""
    try:
        match_keys = schema.list_keys(KeyType.MATCH, 'live')
        
        matches = []
        for key in match_keys:
            match_data = redis_client.get(key)
            if match_data:
                try:
                    matches.append(json.loads(match_data))
                except json.JSONDecodeError:
                    continue
        
        return {
            "matches": matches,
            "count": len(matches),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get live matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/upcoming")
async def get_upcoming_matches(
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get upcoming matches"""
    try:
        match_keys = schema.list_keys(KeyType.MATCH, 'upcoming')
        
        matches = []
        cutoff_time = datetime.now() + timedelta(hours=hours)
        
        for key in match_keys:
            match_data = redis_client.get(key)
            if match_data:
                try:
                    match = json.loads(match_data)
                    
                    # Filter by time if match has start time
                    if 'start_time' in match:
                        start_time = datetime.fromisoformat(match['start_time'])
                        if start_time > cutoff_time:
                            continue
                    
                    matches.append(match)
                except json.JSONDecodeError:
                    continue
        
        # Sort by start time
        matches.sort(key=lambda x: x.get('start_time', ''))
        
        return {
            "matches": matches,
            "count": len(matches),
            "hours_ahead": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get upcoming matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/odds/{match_id}")
async def get_match_odds(
    match_id: str,
    include_history: bool = Query(False),
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get odds for specific match"""
    try:
        # Get current aggregated odds
        current_odds = schema.get_with_json(KeyType.ODDS, 'aggregated', match_id=match_id)
        
        response = {
            "match_id": match_id,
            "current_odds": current_odds,
            "timestamp": datetime.now().isoformat()
        }
        
        if include_history:
            odds_history = schema.get_with_json(KeyType.ODDS, 'history', match_id=match_id) or []
            response["odds_history"] = odds_history
        
        if not current_odds:
            raise HTTPException(status_code=404, detail="Odds not found for this match")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get odds for {match_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/performance")
async def get_performance_metrics(
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get real-time performance metrics"""
    try:
        metrics = schema.get_with_json(KeyType.METRICS, 'performance')
        
        if not metrics:
            return {
                "message": "No performance metrics available",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/{service}")
async def get_service_metrics(
    service: str,
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get metrics for specific service"""
    try:
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Validate service name
        valid_services = ['scraper', 'predictions', 'signals']
        if service not in valid_services:
            raise HTTPException(status_code=400, detail=f"Invalid service. Must be one of: {valid_services}")
        
        metrics = schema.get_with_json(KeyType.METRICS, service, date=date)
        
        if not metrics:
            return {
                "service": service,
                "date": date,
                "message": "No metrics available for this date",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "service": service,
            "date": date,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics for {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/redis/stats")
async def get_redis_stats(
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Get Redis memory usage and statistics"""
    try:
        memory_usage = schema.get_memory_usage()
        queue_stats = schema.get_queue_stats()
        
        # Get additional Redis info
        redis_info = redis_client.info()
        
        return {
            "memory_usage": memory_usage,
            "queue_stats": queue_stats,
            "redis_info": {
                "version": redis_info.get('redis_version'),
                "uptime_seconds": redis_info.get('uptime_in_seconds'),
                "connected_clients": redis_info.get('connected_clients'),
                "total_commands_processed": redis_info.get('total_commands_processed'),
                "keyspace_hits": redis_info.get('keyspace_hits'),
                "keyspace_misses": redis_info.get('keyspace_misses'),
                "used_memory_human": redis_info.get('used_memory_human')
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get Redis stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redis/cleanup")
async def cleanup_redis(
    schema: RedisSchemaManager = Depends(get_redis_manager)
):
    """Clean up expired Redis keys"""
    try:
        cleanup_stats = schema.cleanup_expired()
        
        return {
            "message": "Cleanup completed",
            "cleanup_stats": cleanup_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Redis cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    try:
        # Get system health from Redis
        system_health = schema_manager.get_with_json(KeyType.HEALTH, 'system')
        
        # Get active alerts
        alerts = schema_manager.get_with_json(KeyType.HEALTH, 'alerts') or []
        
        # Count active components
        service_healths = {}
        for service in ['prediction_tracker', 'scraper', 'odds_fetcher']:
            health = await _check_service_health(service)
            service_healths[service] = health.get('status', 'unknown')
        
        active_services = sum(1 for status in service_healths.values() if status == 'healthy')
        total_services = len(service_healths)
        
        overall_status = "healthy" if active_services == total_services else (
            "degraded" if active_services > 0 else "unhealthy"
        )
        
        return {
            "status": overall_status,
            "services": {
                "active": active_services,
                "total": total_services,
                "details": service_healths
            },
            "system_health": system_health,
            "active_alerts": len(alerts),
            "alerts": alerts[-5:] if alerts else [],  # Last 5 alerts
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
