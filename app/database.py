from contextlib import asynccontextmanager
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from .models import Base

# Simple local SQLite DB for tests
DATABASE_URL = "sqlite+aiosqlite:///./scraper_test.db"

_engine = create_async_engine(DATABASE_URL, echo=False, future=True)
_SessionLocal = sessionmaker(bind=_engine, class_=AsyncSession, expire_on_commit=False)

_initialized = False

async def _init_db_once() -> None:
    global _initialized
    if _initialized:
        return
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _initialized = True

@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    await _init_db_once()
    async with _SessionLocal() as session:
        yield session
