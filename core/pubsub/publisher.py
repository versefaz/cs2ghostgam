# core/pubsub/publisher.py
import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class NullPublisher:
    async def publish(self, channel: str, payload: dict):
        return True

    async def close(self):
        return True

    @property
    def mode(self):
        return "null"


async def get_publisher(url: str, enabled: bool, connect_timeout: float = 1.0):
    if not enabled:
        logger.warning("Redis disabled via flag; using NullPublisher")
        return NullPublisher()

    try:
        import aioredis
        redis = await asyncio.wait_for(aioredis.from_url(url), timeout=connect_timeout)
        # sanity ping
        await asyncio.wait_for(redis.ping(), timeout=connect_timeout)
        logger.info("Connected to Redis at %s", url)
        return RedisPublisher(redis)
    except Exception as e:
        logger.warning("Redis unavailable (%s); falling back to NullPublisher", e)
        return NullPublisher()


class RedisPublisher:
    def __init__(self, client):
        self.client = client

    async def publish(self, channel: str, payload: dict):
        import json
        return await self.client.publish(channel, json.dumps(payload))

    async def close(self):
        try:
            await self.client.close()
        except Exception:
            pass

    @property
    def mode(self):
        return "redis"
