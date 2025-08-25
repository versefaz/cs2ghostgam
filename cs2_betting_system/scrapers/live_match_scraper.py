import os
import json
import random
import time
from datetime import datetime
from typing import List, Dict

import redis
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import settings


class LiveMatchScraper:
    def __init__(self):
        self.driver = None
        self.redis_client = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
        self.alerts = AlertSystem()
        self.setup_driver()

    def setup_driver(self):
        """Setup Selenium driver; if fails, leave None and run in degraded mode."""
        try:
            # randomize user agent per session
            ua_pool = [
                settings.USER_AGENT,
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            ]
            ua = random.choice(ua_pool)

            options = Options()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={ua}')

            if settings.USE_UC:
                try:
                    import undetected_chromedriver as uc  # type: ignore
                    self.driver = uc.Chrome(options=options)
                except Exception as e:
                    print(f"UC driver failed, fallback to webdriver.Chrome: {e}")
                    self.driver = webdriver.Chrome(options=options)
            else:
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
            self._human_wait()
            self.driver.get('https://www.hltv.org/matches')
            WebDriverWait(self.driver, settings.SELENIUM_WAIT).until(
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
                    match_data.update(self.get_cached_odds(match_data['team1'], match_data['team2']))
                    matches.append(match_data)
                except Exception:
                    continue

            upcoming = self.driver.find_elements(By.CSS_SELECTOR, '.upcomingMatch')
            for match in upcoming[:20]:
                try:
                    m = {
                        'match_id': match.get_attribute('data-match-id') or f'up_{int(time.time())}',
                        'team1': match.find_element(By.CSS_SELECTOR, '.team1 .teamName').text,
                        'team2': match.find_element(By.CSS_SELECTOR, '.team2 .teamName').text,
                        'status': 'upcoming',
                        'timestamp': datetime.now(),
                    }
                    m.update(self.get_cached_odds(m['team1'], m['team2']))
                    matches.append(m)
                except Exception:
                    continue
        except Exception as e:
            print(f"HLTV scraping error: {e}")
        return matches

    def _search_oddsportal(self, team1: str, team2: str) -> Dict:
        """Navigate OddsPortal search and try to open the match page. Return odds dict or empty if fail."""
        try:
            q = f"{team1} vs {team2}"
            self.driver.get(settings.ODDSPORTAL_SEARCH_URL + q.replace(' ', '+'))
            WebDriverWait(self.driver, settings.SELENIUM_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/matches/"]'))
            )
            # Pick first relevant match link
            links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/matches/"]')
            for a in links:
                href = a.get_attribute('href') or ''
                txt = a.text.lower()
                if team1.lower() in txt and team2.lower() in txt:
                    a.click()
                    break
            return self._extract_odds_from_current_page()
        except Exception:
            return {}

    def _extract_odds_from_current_page(self) -> Dict:
        """Extract odds table and choose the best bookmaker based on BOOKMAKER_PRIORITY."""
        try:
            # Wait for odds table (selectors may vary; keep resilient)
            WebDriverWait(self.driver, settings.SELENIUM_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'table'))
            )
            rows = self.driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')
            best = None
            for row in rows:
                try:
                    bookmaker = row.find_element(By.CSS_SELECTOR, 'td:nth-child(1)').text.strip()
                    # Odds columns might be 2nd and 3rd or similar
                    odds_cells = row.find_elements(By.CSS_SELECTOR, 'td')
                    if len(odds_cells) < 3:
                        continue
                    # Try to parse two runner odds
                    o1 = odds_cells[1].text.strip()
                    o2 = odds_cells[2].text.strip()
                    odds1 = float(o1) if o1 and o1.replace('.', '', 1).isdigit() else None
                    odds2 = float(o2) if o2 and o2.replace('.', '', 1).isdigit() else None
                    if odds1 and odds2:
                        priority = settings.BOOKMAKER_PRIORITY.index(bookmaker) if bookmaker in settings.BOOKMAKER_PRIORITY else 999
                        cand = (priority, bookmaker, odds1, odds2)
                        if best is None or cand < best:
                            best = cand
                except Exception:
                    continue
            if best:
                return {
                    'odds_team1': best[2],
                    'odds_team2': best[3],
                    'odds_source': best[1],
                }
        except Exception:
            pass
        return {}

    def get_live_odds(self, team1: str, team2: str) -> Dict:
        """Try to fetch odds from OddsPortal; return None fields on failure."""
        if not self.driver:
            return {'odds_team1': None, 'odds_team2': None, 'odds_source': None}
        try:
            odds = self._search_oddsportal(team1, team2)
            if odds:
                return odds
        except Exception:
            self.alerts.count_error()
            if self.alerts.should_alert():
                self.alerts.alert_critical(f"Odds scraping failed repeatedly for {team1} vs {team2}")
        return {'odds_team1': None, 'odds_team2': None, 'odds_source': None}

    def get_live_odds_with_protection(self, team1: str, team2: str) -> Dict:
        """Apply anti-bot protections then scrape odds."""
        # Random human-like delay and mouse movement
        self._human_wait()
        try:
            if self.driver:
                actions = ActionChains(self.driver)
                actions.move_by_offset(random.randint(1, 100), random.randint(1, 100)).perform()
        except Exception:
            pass
        return self.get_live_odds(team1, team2)

    def get_cached_odds(self, team1: str, team2: str) -> Dict:
        """Use Redis cache for odds to reduce hits and evade anti-bot."""
        try:
            cache_key = f"odds:{team1}:{team2}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            odds = self.get_live_odds_with_protection(team1, team2)
            if odds and (odds.get('odds_team1') and odds.get('odds_team2')):
                self.redis_client.setex(cache_key, settings.ODDS_CACHE_TTL, json.dumps(odds))
            return odds
        except Exception as e:
            print(f"Cache error: {e}")
            return self.get_live_odds_with_protection(team1, team2)

    def _human_wait(self):
        time.sleep(random.uniform(2, 5))

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


class AlertSystem:
    def __init__(self):
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK') or settings.DISCORD_WEBHOOK
        self.error_count = 0

    def count_error(self):
        self.error_count += 1

    def should_alert(self) -> bool:
        return self.error_count > 5 and bool(self.discord_webhook)

    def alert_critical(self, message: str):
        try:
            if not self.discord_webhook:
                return
            requests.post(self.discord_webhook, json={
                "content": f"🚨 CRITICAL: {message}"
            }, timeout=10)
        except Exception:
            pass
