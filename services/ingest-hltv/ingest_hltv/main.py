import asyncio
import orjson
import httpx
from nats.aio.client import Client as NATS

HLTV_API = "https://api.hltv.org/v1/matches"
NATS_URL = "nats://nats:4222"

async def fetch():
    async with httpx.AsyncClient(timeout=10) as client:
        nc = NATS()
        await nc.connect(servers=[NATS_URL])
        try:
            while True:
                r = await client.get(HLTV_API, params={"startDate": "now", "endDate": "now+1d"})
                r.raise_for_status()
                data = r.json()
                for match in data:
                    await nc.publish("raw.match", orjson.dumps(match))
                await asyncio.sleep(30)
        finally:
            await nc.drain()

if __name__ == "__main__":
    asyncio.run(fetch())
