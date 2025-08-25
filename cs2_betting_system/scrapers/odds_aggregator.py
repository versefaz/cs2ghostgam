from typing import Dict, List, Optional
from .live_match_scraper import LiveMatchScraper

class OddsAggregator:
    """Aggregate odds from scraper (and future sources)."""
    def __init__(self, scraper: Optional[LiveMatchScraper] = None):
        self.scraper = scraper or LiveMatchScraper()

    def get_best_and_average(self, team1: str, team2: str) -> Dict:
        # For now, rely on existing scraper's cached odds
        odds = self.scraper.get_cached_odds(team1, team2)
        if not odds:
            return {'best': {}, 'average': {}, 'sources': [], 'raw': {}}
        # Normalize current scraper output to simple schema
        raw = {'team1': odds.get('odds_team1'), 'team2': odds.get('odds_team2'), 'source': odds.get('odds_source')}
        return {
            'best': {'team1': raw['team1'], 'team2': raw['team2']},
            'average': {'team1': raw['team1'], 'team2': raw['team2']},
            'sources': [raw.get('source')] if raw.get('source') else [],
            'raw': raw,
        }
