import asyncpg
import logging
from typing import Optional
from .config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=2,
            max_size=10,
        )
        await self._ensure_tables()
        logger.info("Database connected")

    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
            logger.info("Database disconnected")

    async def _ensure_tables(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS matches (
                    match_id VARCHAR(50) PRIMARY KEY,
                    team1_id INTEGER,
                    team2_id INTEGER,
                    match_date TIMESTAMPTZ DEFAULT NOW(),
                    map_name VARCHAR(50)
                );
                """
            )
