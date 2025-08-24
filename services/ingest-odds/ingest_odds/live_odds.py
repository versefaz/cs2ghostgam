import aiohttp
import asyncio
import os
from typing import Dict, Any, List, Optional
import nats
import orjson
import redis.asyncio as redis


class LiveOddsStreamer:
    """
    ดึง Odds แบบ realtime จากหลายเว็บพนัน และเผยแพร่ผ่าน NATS + Redis Streams
    Update ทุก 5 วินาที หรือเมื่อมีการเปลี่ยนแปลง
    """

    def __init__(self) -> None:
        self.nats_url = os.getenv("NATS_URL", "nats://nats:4222")
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.match_id_env = int(os.getenv("MATCH_ID", "0"))
        # API keys (optional) per bookmaker
        self.api_keys = {
            "1xbet": os.getenv("ONEXBET_API_KEY"),
            "betway": os.getenv("BETWAY_API_KEY"),
            "pinnacle": os.getenv("PINNACLE_API_KEY"),
        }
        # endpoints (override-able)
        self.endpoints = {
            "1xbet": os.getenv("ONEXBET_API", "https://1xbet.com/LiveFeed/Get1x2_VZip"),
            "betway": os.getenv("BETWAY_API", "https://api.betway.com/v2/live/esports"),
            "pinnacle": os.getenv("PINNACLE_API", "https://api.pinnacle.com/v1/odds"),
        }
        self._nc: Optional[nats.NATS] = None
        self._redis: Optional[redis.Redis] = None

    async def _ensure_clients(self) -> None:
        if not self._nc:
            self._nc = await nats.connect(servers=[self.nats_url])
        if not self._redis:
            self._redis = redis.from_url(self.redis_url)

    async def stream_odds(self, match_id: Optional[int] = None):
        mid = match_id if match_id is not None else self.match_id_env
        await self._ensure_clients()
        assert self._nc and self._redis
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = [self.fetch_odds(session, bk, mid) for bk in self.endpoints.keys()]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                cleaned = [r for r in results if not isinstance(r, Exception) and r is not None]
                if cleaned:
                    await self.publish_odds_update(mid, cleaned)
                await asyncio.sleep(int(os.getenv("ODDS_INTERVAL", "5")))

    async def fetch_odds(self, session: aiohttp.ClientSession, bookmaker: str, match_id: int) -> Optional[Dict[str, Any]]:
        url = self.endpoints[bookmaker]
        key = self.api_keys.get(bookmaker)
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        params = {"matchId": match_id}
        try:
            async with session.get(url, headers=headers, params=params, timeout=15) as resp:
                resp.raise_for_status()
                data = await resp.json()
                # normalize to a simple structure per market/selection if possible
                return {"bookmaker": bookmaker, "match_id": match_id, "raw": data}
        except Exception:
            return None

    async def publish_odds_update(self, match_id: int, results: List[Dict[str, Any]]):
        assert self._nc and self._redis
        subject = f"live.odds.{match_id}"
        payload = {"match_id": match_id, "results": results}
        encoded = orjson.dumps(payload)
        # Publish to NATS
        await self._nc.publish(subject, encoded)
        # Mirror to Redis Stream for GraphQL subscription
        await self._redis.xadd(
            f"live:odds:{match_id}",
            {b"payload": encoded},
        )
