from typing import List, Dict
import random
from datetime import datetime


BOOKMAKERS = [
    "Pinnacle",
    "Bet365",
    "1xBet",
    "GG.BET",
    "Unikrn",
]


class OddsScraper:
    async def scrape_cs2_odds(self) -> List[Dict]:
        """
        Return a small set of realistic mock odds. In production, replace with
        integrations to bookmaker APIs or HTML scrapers.
        """
        odds: List[Dict] = []
        now = datetime.utcnow()
        # Generate 5 odds entries
        for i in range(5):
            bm = random.choice(BOOKMAKERS)
            # Ensure implied probability stays in a reasonable range
            team1 = round(random.uniform(1.2, 4.5), 2)
            team2 = round(random.uniform(1.2, 4.5), 2)
            odds.append({
                "match_id": f"mock-{now.strftime('%Y%m%d%H%M')}-{i}",
                "bookmaker": bm,
                "team1_odds": team1,
                "team2_odds": team2,
            })
        return odds
