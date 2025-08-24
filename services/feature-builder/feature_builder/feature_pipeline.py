from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime
import asyncio
import orjson

from .database import DatabaseManager
from .redis_client import RedisFeatureStore
from .feature_calculators import TeamFeatureCalculator, PlayerFeatureCalculator, MatchContextCalculator

class FeaturePipeline:
    def __init__(self, db: DatabaseManager, redis_store: RedisFeatureStore) -> None:
        self.db = db
        self.redis = redis_store
        self.team_calc = TeamFeatureCalculator(db)
        self.player_calc = PlayerFeatureCalculator(db)
        self.context_calc = MatchContextCalculator(db)

    async def get_match_features(self, match_id: str, team1_id: int, team2_id: int, map_name: str, force_refresh: bool = False) -> Dict[str, Any]:
        # For now, compute fresh on every call
        t1 = await self.team_calc.calculate_team_features(team1_id, map_name)
        t2 = await self.team_calc.calculate_team_features(team2_id, map_name)
        h2h = await self.team_calc.calculate_h2h_features(team1_id, team2_id)
        ctx = await self.context_calc.calculate_match_context(match_id)
        features = {
            "team1": t1,
            "team2": t2,
            "h2h": h2h,
            "context": ctx,
            "calc_at": datetime.utcnow().isoformat(),
        }
        await self.redis.setex(f"features:match:{match_id}", 3600, orjson.dumps(features))
        return features

    async def get_team_features(self, team_id: int, days_back: int = 90) -> Dict[str, Any]:
        return await self.team_calc.calculate_team_features(team_id)

    async def get_player_features(self, player_id: int, days_back: int = 30) -> Dict[str, Any]:
        return await self.player_calc.calculate_player_features(player_id)

    async def get_batch_features(self, match_ids: List[str]) -> List[Dict[str, Any]]:
        return [{"match_id": mid, "features": None} for mid in match_ids]

    async def refresh_all_features(self) -> None:
        # Placeholder: iterate over keys and recompute
        pass

    async def start_feature_refresh_worker(self) -> None:
        interval = 300
        while True:
            await asyncio.sleep(interval)
            try:
                await self.refresh_all_features()
            except Exception:
                pass
