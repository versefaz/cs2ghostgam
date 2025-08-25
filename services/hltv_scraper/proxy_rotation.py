import asyncio
import random
from typing import Optional, Dict
from config import Settings

class ProxyRotator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._proxies = []  # replace with DB/Redis if needed
        self._idx = 0

    async def load_proxies(self):
        # Placeholder: load from settings/env/redis/db
        self._proxies = [
            {"host": "proxy", "port": 8080},
        ] if self.settings.proxy_enabled else []

    async def get_proxy(self) -> Optional[Dict]:
        if not self._proxies:
            await self.load_proxies()
        if not self._proxies:
            return None
        self._idx = (self._idx + 1) % len(self._proxies)
        return self._proxies[self._idx]
