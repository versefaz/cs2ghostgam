import aiohttp
import asyncio
from typing import Dict, Any, List

class LiveOddsStreamer:
    """
    ดึง Odds แบบ realtime จากหลายเว็บพนัน
    Update ทุก 5 วินาที หรือเมื่อมีการเปลี่ยนแปลง
    """
    BOOKMAKERS: Dict[str, Dict[str, str]] = {
        "1xbet": {"api": "https://1xbet.com/LiveFeed/Get1x2_VZip", "key": "YOUR_API_KEY"},
        "betway": {"api": "https://api.betway.com/v2/live/esports", "key": "YOUR_API_KEY"},
        "pinnacle": {"api": "https://api.pinnacle.com/v1/odds", "key": "YOUR_API_KEY"},
    }

    async def stream_odds(self, match_id: int):
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = [self.fetch_odds(session, bk, cfg, match_id) for bk, cfg in self.BOOKMAKERS.items()]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                await self.publish_odds_update([r for r in results if not isinstance(r, Exception)])
                await asyncio.sleep(5)

    async def fetch_odds(self, session: aiohttp.ClientSession, bookmaker: str, cfg: Dict[str, str], match_id: int) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {cfg['key']}"}
        params = {"matchId": match_id}
        async with session.get(cfg["api"], headers=headers, params=params, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"bookmaker": bookmaker, "data": data}

    async def publish_odds_update(self, results: List[Dict[str, Any]]):
        # TODO: publish to NATS/Kafka topic raw.odds
        _ = results
        return
