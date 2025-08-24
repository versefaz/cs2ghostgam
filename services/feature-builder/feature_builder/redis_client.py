import redis.asyncio as redis
import logging
from typing import Optional, Dict, Any
from .config import settings

logger = logging.getLogger(__name__)

class RedisFeatureStore:
    def __init__(self) -> None:
        self.client: Optional[redis.Redis] = None
        self.default_ttl = settings.feature_ttl

    async def connect(self) -> None:
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=False,
        )
        await self.client.ping()
        logger.info("Redis connected")

    async def disconnect(self) -> None:
        if self.client:
            await self.client.close()

    async def setex(self, key: str, ttl: int, value: bytes) -> None:
        await self.client.setex(key, ttl, value)

    async def get(self, key: str) -> Optional[bytes]:
        return await self.client.get(key)

    async def publish(self, channel: str, data: bytes) -> None:
        await self.client.publish(channel, data)
