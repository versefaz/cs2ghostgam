import asyncio
from datetime import datetime
from typing import List, Dict
import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

class HLTVScraper:
    async def scrape_upcoming_matches(self) -> List[Dict]:
        url = "https://www.hltv.org/matches"
        async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'lxml')
            out: List[Dict] = []
            for el in soup.select('.upcomingMatch')[:50]:
                try:
                    mid = el.get('data-zonedgrouping-entry-unix') or el.get('data-match-id') or str(hash(el.text))
                    t1 = el.select_one('.team1 .matchTeamName') or el.select_one('.team1 .teamName')
                    t2 = el.select_one('.team2 .matchTeamName') or el.select_one('.team2 .teamName')
                    time_el = el.select_one('.matchTime')
                    ev = el.select_one('.matchEventName')
                    if not t1 or not t2:
                        continue
                    out.append({
                        'match_id': mid,
                        'team1_name': t1.text.strip(),
                        'team2_name': t2.text.strip(),
                        'match_time': datetime.utcnow(),
                        'event_name': ev.text.strip() if ev else None,
                    })
                except Exception:
                    continue
            return out

    async def scrape_top_players(self) -> List[Dict]:
        url = "https://www.hltv.org/stats/players?startDate=all"
        async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'lxml')
            out: List[Dict] = []
            for row in soup.select('table.stats-table tbody tr')[:50]:
                try:
                    link = row.select_one('td.playerCol a')
                    name = row.select_one('td.playerCol a').text.strip()
                    rating = float(row.select_one('td.ratingCol').text.strip())
                    kd = float(row.select_one('td.kdCol').text.strip())
                    pid = link['href'].strip('/').split('/')[-2] if link else name
                    out.append({
                        'player_id': pid,
                        'name': name,
                        'rating': rating,
                        'kd_ratio': kd,
                    })
                except Exception:
                    continue
            return out
