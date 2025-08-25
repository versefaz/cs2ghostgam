from typing import Dict, List

class HLTVMatchScraper:
    async def run(self) -> List[Dict]:
        return []

class HLTVTeamScraper:
    async def run(self, team_id: str) -> Dict:
        return {}

class HLTVPlayerScraper:
    async def run(self, player_id: str) -> Dict:
        return {}
