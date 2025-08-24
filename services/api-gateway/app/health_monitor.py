import asyncio
import logging
from datetime import datetime
from typing import Any

from .config import config

logger = logging.getLogger(__name__)

class HealthMonitor:
    def __init__(self, client: Any):
        self.client = client

    async def start(self):
        while True:
            await asyncio.sleep(config.HEALTH_CHECK_INTERVAL)
            logger.info(f"Health check at {datetime.now().isoformat()} connected={self.client.connected}")
