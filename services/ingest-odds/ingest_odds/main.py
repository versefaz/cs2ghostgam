import asyncio
import os
from .live_odds import LiveOddsStreamer


async def amain() -> None:
    match_id = int(os.getenv("MATCH_ID", "0"))
    streamer = LiveOddsStreamer()
    await streamer.stream_odds(match_id)


if __name__ == "__main__":
    asyncio.run(amain())
