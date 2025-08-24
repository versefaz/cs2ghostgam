import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RedisPublisher:
    """Stub async Redis publisher (no-op). Replace with real Redis client if needed."""

    async def publish_system_event(self, payload: Dict[str, Any]):
        logger.info(f"System event: {payload}")

    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        logger.debug(f"Publish {event_type}: {payload}")

    async def close(self):
        return
