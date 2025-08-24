import asyncio
import orjson
import redis
from nats.aio.client import Client as NATS
from feature_builder.pipeline import build_vector

REDIS_HOST = "redis"
NATS_URL = "nats://nats:4222"

async def run():
    rds = redis.Redis(host=REDIS_HOST)
    nc = NATS()
    await nc.connect(servers=[NATS_URL])

    async def handler(msg):
        data = orjson.loads(msg.data)
        match_id = data.get("id")
        # TODO: enrich from Postgres & odds service
        team_stats = {"elo_a": 1700, "elo_b": 1650, "form_a10": 0.6}
        player_stats = {}
        odds = {"moneyline_a": 1.85}
        vec = build_vector(team_stats, player_stats, odds)
        rds.setex(f"feature:{match_id}", 6*60*60, orjson.dumps(vec))

    await nc.subscribe("raw.match", cb=handler)
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await nc.drain()

if __name__ == "__main__":
    asyncio.run(run())
