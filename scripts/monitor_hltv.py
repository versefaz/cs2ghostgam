import asyncio
from playwright.async_api import async_playwright


async def monitor_live_matches(duration_ms: int = 60_000) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        def on_ws(ws):
            print(f"[WS] {ws.url}")
            ws.on("framereceived", lambda msg: print(f"[WS FRAME] {str(msg)[:200]}..."))

        page.on("websocket", on_ws)
        await page.goto("https://www.hltv.org/matches")
        await page.wait_for_timeout(duration_ms)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(monitor_live_matches())
