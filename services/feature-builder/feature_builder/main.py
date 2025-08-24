from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import logging
import orjson
import redis
from nats.aio.client import Client as NATS
from feature_builder.pipeline import build_vector
from feature_builder.config import settings
from feature_builder.database import DatabaseManager
from feature_builder.redis_client import RedisFeatureStore
from feature_builder.feature_pipeline import FeaturePipeline
from feature_builder.models import FeatureRequest, FeatureResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = "redis"
NATS_URL = "nats://nats:4222"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Feature Builder Service...")
    app.state.db = DatabaseManager()
    app.state.redis = RedisFeatureStore()
    app.state.pipeline = FeaturePipeline(app.state.db, app.state.redis)
    app.state.nc = NATS()
    await app.state.nc.connect(servers=[NATS_URL])

    await app.state.db.connect()
    await app.state.redis.connect()

    # background refresh worker
    app.state.refresh_task = asyncio.create_task(app.state.pipeline.start_feature_refresh_worker())
    try:
        yield
    finally:
        logger.info("Shutting down Feature Builder Service...")
        try:
            app.state.refresh_task.cancel()
        except Exception:
            pass
        await app.state.redis.disconnect()
        await app.state.db.disconnect()
        await app.state.nc.drain()


app = FastAPI(title="CS2 Feature Builder Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "feature-builder"}


@app.post("/features/match", response_model=FeatureResponse)
async def get_match_features(request: FeatureRequest):
    try:
        features = await app.state.pipeline.get_match_features(
            match_id=request.match_id,
            team1_id=request.team1_id,
            team2_id=request.team2_id,
            map_name=request.map_name,
            force_refresh=request.force_refresh,
        )
        return FeatureResponse(match_id=request.match_id, features=features, timestamp=datetime.utcnow())
    except Exception as e:
        logger.exception("Error getting match features")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/team/{team_id}")
async def get_team_features(team_id: int, days_back: int = 90):
    features = await app.state.pipeline.get_team_features(team_id, days_back)
    return {"team_id": team_id, "features": features}


@app.get("/features/player/{player_id}")
async def get_player_features(player_id: int, days_back: int = 30):
    features = await app.state.pipeline.get_player_features(player_id, days_back)
    return {"player_id": player_id, "features": features}


@app.post("/features/batch")
async def get_batch_features(match_ids: list[str]):
    results = await app.state.pipeline.get_batch_features(match_ids)
    return {"matches": results}


@app.post("/features/refresh")
async def refresh_features(background_tasks: BackgroundTasks):
    background_tasks.add_task(app.state.pipeline.refresh_all_features)
    return {"status": "refresh started"}


@app.on_event("shutdown")
async def shutdown_event():
    await app.state.nc.close()


@app.on_event("startup")
async def startup_event():
    async def handler(msg):
        data = orjson.loads(msg.data)
        match_id = data.get("id")
        # TODO: enrich from Postgres & odds service
        team_stats = {"elo_a": 1700, "elo_b": 1650, "form_a10": 0.6}
        player_stats = {}
        odds = {"moneyline_a": 1.85}
        vec = build_vector(team_stats, player_stats, odds)
        await app.state.redis.setex(f"feature:{match_id}", 6*60*60, orjson.dumps(vec))

    await app.state.nc.subscribe("raw.match", cb=handler)
