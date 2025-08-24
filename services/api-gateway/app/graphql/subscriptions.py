import asyncio
from typing import AsyncGenerator
import strawberry
import json

# NOTE: Wire real clients later (DI or app.state)
try:
    import redis.asyncio as redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

try:
    import nats  # type: ignore
except Exception:  # pragma: no cover
    nats = None  # type: ignore

@strawberry.type
class OddsUpdate:
    market: str
    selection: str
    odds: float
    movement: float | None
    volume: float | None
    timestamp: str

@strawberry.type
class MatchUpdate:
    match_id: str
    team1_score: int | None
    team2_score: int | None
    round: int | None
    timestamp: str | None

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def live_odds(self, match_id: str) -> AsyncGenerator[OddsUpdate, None]:
        if redis is None:
            # fallback: empty stream
            while True:
                await asyncio.sleep(5)
        r = redis.Redis()
        stream = f"live:odds:{match_id}"
        last_id = "$"
        while True:
            resp = await r.xread({stream: last_id}, block=15000, count=1)
            if not resp:
                continue
            _, entries = resp[0]
            last_id, fields = entries[0]
            payload = json.loads(fields.get(b"payload", b"{}"))
            yield OddsUpdate(**payload)

    @strawberry.subscription
    async def match_updates(self, match_id: str) -> AsyncGenerator[MatchUpdate, None]:
        if nats is None:
            while True:
                await asyncio.sleep(5)
        nc = await nats.connect(servers=["nats://nats:4222"])  # TODO: env
        sub = await nc.subscribe(f"live.match.{match_id}")
        try:
            async for msg in sub.messages:  # type: ignore[attr-defined]
                data = json.loads(msg.data)
                yield MatchUpdate(
                    match_id=str(data.get("match_id")),
                    team1_score=data.get("team1_score"),
                    team2_score=data.get("team2_score"),
                    round=data.get("round"),
                    timestamp=str(data.get("timestamp")),
                )
        finally:
            await sub.unsubscribe()  # type: ignore
            await nc.drain()
