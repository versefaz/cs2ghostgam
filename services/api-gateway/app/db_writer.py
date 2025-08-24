import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DatabaseWriter:
    """Stub async DB writer (no-op). Replace with real asyncpg/SQLAlchemy writer."""

    async def write_event(self, event_type: str, payload: Dict[str, Any]):
        logger.debug(f"Write {event_type} to DB: {payload}")

    async def close(self):
        return
