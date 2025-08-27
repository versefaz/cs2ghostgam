#!/usr/bin/env python3
"""
Session Manager - จัดการ aiohttp sessions แบบ singleton pattern
แก้ปัญหา Unclosed Client Sessions
"""

import asyncio
import aiohttp
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """จัดการ aiohttp sessions แบบ singleton pattern"""
    
    _instance = None
    _sessions: Dict[str, aiohttp.ClientSession] = {}
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_session(self, name: str, **kwargs) -> aiohttp.ClientSession:
        """สร้างหรือคืน session ที่มีอยู่แล้ว"""
        async with self._lock:
            if name not in self._sessions or self._sessions[name].closed:
                # Default connector settings
                connector = aiohttp.TCPConnector(
                    limit=kwargs.get('limit', 3),  # จำกัด connections
                    limit_per_host=kwargs.get('limit_per_host', 2),
                    ttl_dns_cache=300,
                    use_dns_cache=True
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=kwargs.get('timeout', 30),
                    connect=kwargs.get('connect_timeout', 5)
                )
                
                headers = kwargs.get('headers', {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                self._sessions[name] = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers=headers
                )
                logger.debug(f"Created new session: {name}")
            
            return self._sessions[name]
    
    async def close_session(self, name: str):
        """ปิด session เฉพาะ"""
        async with self._lock:
            if name in self._sessions and not self._sessions[name].closed:
                await self._sessions[name].close()
                del self._sessions[name]
                logger.debug(f"Closed session: {name}")
    
    async def close_all(self):
        """ปิดทุก sessions อย่างปลอดภัย"""
        async with self._lock:
            for name, session in list(self._sessions.items()):
                if not session.closed:
                    try:
                        await session.close()
                        logger.debug(f"Closed session: {name}")
                    except Exception as e:
                        logger.warning(f"Error closing session {name}: {e}")
            
            self._sessions.clear()
            logger.info("All HTTP sessions closed")
    
    def get_active_sessions(self) -> Dict[str, bool]:
        """ดูสถานะ sessions ที่ใช้งานอยู่"""
        return {
            name: not session.closed 
            for name, session in self._sessions.items()
        }


# Singleton instance
session_manager = SessionManager()
