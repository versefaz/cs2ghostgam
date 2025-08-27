# core/session/session_manager.py
import aiohttp
import asyncio
import atexit
import logging
import weakref
from typing import Optional

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self._base_session: Optional[aiohttp.ClientSession] = None
        self._children = weakref.WeakSet()
        self._lock = asyncio.Lock()

    async def get_session(self, headers=None, timeout=15):
        async with self._lock:
            if self._base_session and not self._base_session.closed:
                return self._base_session
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            self._base_session = aiohttp.ClientSession(timeout=timeout_obj, headers=headers)
            return self._base_session

    async def create_child_session(self, headers=None, timeout=15):
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        s = aiohttp.ClientSession(timeout=timeout_obj, headers=headers)
        self._children.add(s)
        return s

    async def close_all(self):
        tasks = []
        if self._base_session and not self._base_session.closed:
            tasks.append(self._base_session.close())
        for s in list(self._children):
            if not s.closed:
                tasks.append(s.close())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("SessionManager closed all sessions")


session_manager = SessionManager()


def _sync_close():
    loop = None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(session_manager.close_all())
        else:
            loop.run_until_complete(session_manager.close_all())
    except Exception:
        pass


atexit.register(_sync_close)
