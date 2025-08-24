import asyncio
from typing import Any, Dict, Optional
from playwright.async_api import async_playwright
import redis.asyncio as redis
import orjson
import json
import nats

class HLTVRealtimeScraper:
    """
    ดึงข้อมูลแบบ realtime จาก HLTV Live Score
    - Match live score updates ทุก 1 วินาที
    - Round-by-round economy data
    - Player stats แบบ tick-by-tick
    """
    def __init__(self) -> None:
        self.redis = redis.Redis(decode_responses=False)
        self.ws_url = "wss://scorebot-secure.hltv.org/socket.io/"
        self.nc: Optional[nats.NATS] = None

    async def connect_live_match(self, match_id: int):
        """เชื่อมต่อ WebSocket สำหรับแมตช์ที่กำลังแข่ง (skeleton)"""
        # connect NATS once
        if not self.nc:
            self.nc = await nats.connect(servers=["nats://nats:4222"])  # override via env later
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # NOTE: Playwright WebSocket API needs explicit listeners per socket
            def on_ws(ws):
                ws.on("framereceived", lambda msg: asyncio.create_task(self.handle_ws_message(match_id, msg)))
            page.on("websocket", on_ws)

            await page.goto(f"https://www.hltv.org/matches/{match_id}")
            # Keep page open
            while True:
                await asyncio.sleep(1)

    async def handle_ws_message(self, match_id: int, message: str):
        """ประมวลผล live data และส่งต่อทันที"""
        try:
            data = orjson.loads(message)
        except Exception:
            return
        # parse and publish to NATS
        parsed = await self.parse_hltv_message(data)
        if parsed and self.nc:
            await self.publish_to_nats(parsed)
        await self.redis.xadd(
            f"live:match:{match_id}",
            {
                b"round": orjson.dumps(data.get("round")),
                b"score_ct": orjson.dumps(data.get("scoreCT")),
                b"score_t": orjson.dumps(data.get("scoreT")),
                b"economy_ct": orjson.dumps(data.get("economyCT")),
                b"economy_t": orjson.dumps(data.get("economyT")),
                b"timestamp": orjson.dumps(data.get("timestamp")),
            },
        )

    async def parse_hltv_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse HLTV WebSocket messages (sprint 1 minimal)."""
        try:
            mtype = message.get("type")
            if mtype == "scoreUpdate":
                return {
                    "match_id": message.get("matchId"),
                    "team1_score": message.get("score", {}).get("team1"),
                    "team2_score": message.get("score", {}).get("team2"),
                    "round": message.get("round"),
                    "timestamp": message.get("timestamp"),
                }
        except Exception:
            return None
        return None

    async def publish_to_nats(self, data: Dict[str, Any]) -> None:
        """Publish parsed update to NATS."""
        assert self.nc
        subject = f"live.match.{data['match_id']}"
        await self.nc.publish(subject, json.dumps(data, default=str).encode("utf-8"))

if __name__ == "__main__":
    scraper = HLTVRealtimeScraper()
    asyncio.run(scraper.connect_live_match(0))
