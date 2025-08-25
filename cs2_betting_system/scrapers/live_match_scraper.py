import time
from datetime import datetime
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import settings


class LiveMatchScraper:
    def __init__(self):
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Setup Selenium driver; if fails, leave None and run in degraded mode."""
        try:
            options = Options()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={settings.USER_AGENT}')
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            print(f"Selenium unavailable, fallback to mock scraping: {e}")
            self.driver = None

    def scrape_hltv_matches(self) -> List[Dict]:
        """Scrape live/upcoming matches from HLTV. Degrades to empty list if no driver."""
        if not self.driver:
            return []
        matches: List[Dict] = []
        try:
            self.driver.get('https://www.hltv.org/matches')
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'match-day'))
            )

            live_matches = self.driver.find_elements(By.CSS_SELECTOR, '.liveMatch-container')
            for match in live_matches:
                try:
                    match_data = {
                        'match_id': match.get_attribute('data-match-id') or f'live_{int(time.time())}',
                        'team1': match.find_element(By.CSS_SELECTOR, '.team1 .teamName').text,
                        'team2': match.find_element(By.CSS_SELECTOR, '.team2 .teamName').text,
                        'status': 'live',
                        'timestamp': datetime.now(),
                    }
                    match_data.update(self.get_live_odds(match_data['match_id']))
                    matches.append(match_data)
                except Exception:
                    continue

            upcoming = self.driver.find_elements(By.CSS_SELECTOR, '.upcomingMatch')
            for match in upcoming[:20]:
                try:
                    matches.append({
                        'match_id': match.get_attribute('data-match-id') or f'up_{int(time.time())}',
                        'team1': match.find_element(By.CSS_SELECTOR, '.team1 .teamName').text,
                        'team2': match.find_element(By.CSS_SELECTOR, '.team2 .teamName').text,
                        'status': 'upcoming',
                        'timestamp': datetime.now(),
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"HLTV scraping error: {e}")
        return matches

    def get_live_odds(self, match_id: str) -> Dict:
        """Placeholder odds scraping. Return None odds if not available."""
        odds = {
            'odds_team1': None,
            'odds_team2': None,
            'odds_source': None,
        }
        # Real implementation would scrape oddsportal or bookmaker APIs.
        return odds

    def scrape_all_sources(self) -> List[Dict]:
        matches = []
        matches.extend(self.scrape_hltv_matches())
        # Add more sources if desired
        return matches

    def cleanup(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
