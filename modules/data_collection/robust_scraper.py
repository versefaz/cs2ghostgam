import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

# aiohttp import with fallback handling
try:
    import aiohttp
except ImportError:
    aiohttp = None

# Optional heavy deps guarded
try:
    from selenium.webdriver.common.by import By  # type: ignore
    from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
    from selenium.webdriver.support import expected_conditions as EC  # type: ignore
    import undetected_chromedriver as uc  # type: ignore
    SELENIUM_AVAILABLE = True
except Exception:  # pragma: no cover
    SELENIUM_AVAILABLE = False

try:
    from playwright.async_api import async_playwright  # type: ignore
    PLAYWRIGHT_AVAILABLE = True
except Exception:  # pragma: no cover
    PLAYWRIGHT_AVAILABLE = False

try:
    import cloudscraper  # type: ignore
    CLOUDSCRAPER_AVAILABLE = True
except Exception:  # pragma: no cover
    CLOUDSCRAPER_AVAILABLE = False

logger = logging.getLogger(__name__)


def _ensure_aiohttp() -> bool:
    """Check if aiohttp is available for use"""
    if aiohttp is None:
        logger.error("aiohttp not available – falling back to empty result")
        return False
    return True


def _random_ua() -> str:
    """Generate random user agent string"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]
    return random.choice(user_agents)


@dataclass
class MatchOdds:
    team1: str
    team2: str
    odds1: float
    odds2: float
    source: str
    timestamp: str


class RobustOddsScraper:
    """Robust multi-backend scraper with graceful fallbacks and selector retries."""

    def __init__(self):
        self.sources = {
            'pinnacle': {
                'url': 'https://www.pinnacle.com/en/esports/csgo/',
                'selectors': {
                    'match': ['div[data-test-id="event-row"]', '.event-row', '[class*="event"]'],
                    'team': ['.participant-name', '[data-test="competitor-name"]', '.team-name'],
                    'odds': ['.odds-decimal', '[data-test="odds"]', '.price'],
                },
                'method': 'selenium',
            },
            'bet365': {
                'url': 'https://www.bet365.com/#/AC/B151/C1/D43/E0/F163/',
                'selectors': {
                    'match': ['.rcl-ParticipantFixtureDetails', '.sl-MarketCouponFixtureLabelBase'],
                    'team': ['.rcl-ParticipantFixtureDetailsTeam_TeamName', '.sl-CouponParticipantWithBookCloses_Name'],
                    'odds': ['.sgl-ParticipantOddsOnly80_Odds', '.sl-MarketCouponValuesExplicit2_Odds'],
                },
                'method': 'playwright',
            },
            'ggbet': {
                'url': 'https://gg.bet/en/esports/counter-strike',
                'api_endpoint': 'https://gg.bet/api/v1/matches',
                'method': 'api',
            },
            'betway': {
                'url': 'https://betway.com/en/sports/grp/esports/counter-strike',
                'selectors': {
                    'match': ['.event-list__item', '.collapsible-event-markets'],
                    'team': ['.event-list__participant', '.participant-name'],
                    'odds': ['.odds__odd', '.betbutton__odds'],
                },
                'method': 'cloudscraper',
            },
            'rivalry': {
                'url': 'https://www.rivalry.com/esports/csgo-betting',
                'selectors': {
                    'match': ['div[data-testid="match-card"]', '.match-card'],
                    'team': ['.team-name', 'span[data-testid="team-name"]'],
                    'odds': ['.odds-value', 'button[data-testid="odds-button"]'],
                },
                'method': 'selenium',
            },
        }
        self.fallback_strategies = ['selenium', 'playwright', 'cloudscraper', 'api']

    async def scrape_all_sources(self) -> Dict[str, List[Dict]]:
        all_odds: Dict[str, List[Dict]] = {}
        tasks = [self.scrape_with_fallback(name, cfg) for name, cfg in self.sources.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (name, _), res in zip(self.sources.items(), results):
            if isinstance(res, Exception):
                all_odds[name] = await self.emergency_scrape(name)
            else:
                all_odds[name] = res
        return all_odds

    async def scrape_with_fallback(self, source_name: str, config: Dict) -> List[Dict]:
        primary = config.get('method')
        try:
            return await self._run_method(primary, config, source_name)
        except Exception:
            # try fallbacks
            for m in self.fallback_strategies:
                if m == primary:
                    continue
                try:
                    return await self._run_method(m, config, source_name)
                except Exception:
                    continue
        return []

    async def _run_method(self, method: str, config: Dict, source_name: str) -> List[Dict]:
        if method == 'selenium':
            return await self.scrape_with_selenium(config, source_name)
        if method == 'playwright':
            return await self.scrape_with_playwright(config, source_name)
        if method == 'cloudscraper':
            return await self.scrape_with_cloudscraper(config, source_name)
        if method == 'api':
            return await self.scrape_with_api(config, source_name)
        return []

    async def scrape_with_selenium(self, config: Dict, source_name: str) -> List[Dict]:
        if not SELENIUM_AVAILABLE:
            return []
        options = uc.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--headless=new')
        options.add_argument('--user-agent=Mozilla/5.0')
        driver = uc.Chrome(options=options)
        try:
            driver.get(config['url'])
            wait = WebDriverWait(driver, 20)
            matches: List[Dict] = []
            for match_sel in config.get('selectors', {}).get('match', []):
                try:
                    elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, match_sel)))
                except Exception:
                    continue
                if not elements:
                    continue
                for el in elements:
                    md = self._extract_match_data_selenium(el, config.get('selectors', {}))
                    if md:
                        md['source'] = source_name
                        matches.append(md)
                if matches:
                    break
            return matches
        finally:
            try:
                driver.quit()
            except Exception:
                pass

    async def scrape_with_playwright(self, config: Dict, source_name: str) -> List[Dict]:
        if not PLAYWRIGHT_AVAILABLE:
            return []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
            context = await browser.new_context(user_agent='Mozilla/5.0', viewport={'width': 1280, 'height': 800})
            page = await context.new_page()
            await page.goto(config['url'], wait_until='networkidle')
            matches: List[Dict] = []
            for match_sel in config.get('selectors', {}).get('match', []):
                els = await page.query_selector_all(match_sel)
                if not els:
                    continue
                for el in els:
                    md = await self._extract_match_data_playwright(el, config.get('selectors', {}))
                    if md:
                        md['source'] = source_name
                        matches.append(md)
                if matches:
                    break
            await browser.close()
            return matches

    async def scrape_with_cloudscraper(self, config: Dict, source_name: str) -> List[Dict]:
        if not CLOUDSCRAPER_AVAILABLE:
            return []
        scraper = cloudscraper.create_scraper()
        try:
            resp = scraper.get(config['url'], headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code != 200:
                return []
            html = resp.text
            # Minimal parse: as cloudscraper returns static HTML without DOM lib here; skip DOM parse
            # Return empty to keep robust.
            return []
        except Exception:
            return []

    async def scrape_with_api(self, config: Dict, source_name: str) -> List[Dict]:
        """
        ดึงราคาจาก REST-API ของ bookmaker
        Returns: list ของ dict ราคา, ถ้า error → []
        """
        if not _ensure_aiohttp():
            return []
        
        endpoint = config.get('api_endpoint')
        if not endpoint:
            return []
            
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {"User-Agent": _random_ua()}
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(endpoint, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning("API %s status %s", endpoint, resp.status)
                        return []
                    data = await resp.json()
                    return self._extract_from_api_json(data, source_name)
        except Exception as exc:
            logger.warning("scrape_with_api(%s) failed: %s", endpoint, exc)
            return []

    async def emergency_scrape(self, source_name: str) -> List[Dict]:
        # Final fallback: return empty list to avoid crashing pipelines
        return []

    def _extract_match_data_selenium(self, element, selectors: Dict) -> Optional[Dict]:
        md: Dict = {}
        # teams
        for sel in selectors.get('team', []):
            try:
                ts = element.find_elements(By.CSS_SELECTOR, sel)
                if len(ts) >= 2:
                    md['team1'] = ts[0].text.strip()
                    md['team2'] = ts[1].text.strip()
                    break
            except Exception:
                continue
        # odds
        for sel in selectors.get('odds', []):
            try:
                os = element.find_elements(By.CSS_SELECTOR, sel)
                if len(os) >= 2:
                    md['odds1'] = self._to_float(os[0].text)
                    md['odds2'] = self._to_float(os[1].text)
                    break
            except Exception:
                continue
        if {'team1', 'team2', 'odds1', 'odds2'} <= md.keys():
            md['timestamp'] = datetime.utcnow().isoformat()
            return md
        return None

    async def _extract_match_data_playwright(self, element, selectors: Dict) -> Optional[Dict]:
        md: Dict = {}
        for sel in selectors.get('team', []):
            try:
                ts = await element.query_selector_all(sel)
                if len(ts) >= 2:
                    md['team1'] = (await ts[0].inner_text()).strip()
                    md['team2'] = (await ts[1].inner_text()).strip()
                    break
            except Exception:
                continue
        for sel in selectors.get('odds', []):
            try:
                os = await element.query_selector_all(sel)
                if len(os) >= 2:
                    md['odds1'] = self._to_float(await os[0].inner_text())
                    md['odds2'] = self._to_float(await os[1].inner_text())
                    break
            except Exception:
                continue
        if {'team1', 'team2', 'odds1', 'odds2'} <= md.keys():
            md['timestamp'] = datetime.utcnow().isoformat()
            return md
        return None

    def _extract_from_api_json(self, data, source_name: str) -> List[Dict]:
        results: List[Dict] = []
        try:
            for item in (data or []):
                t1 = item.get('team1') or item.get('homeTeam') or ''
                t2 = item.get('team2') or item.get('awayTeam') or ''
                o1 = item.get('odds1') or item.get('homeOdds') or None
                o2 = item.get('odds2') or item.get('awayOdds') or None
                if t1 and t2 and o1 and o2:
                    results.append({
                        'team1': str(t1),
                        'team2': str(t2),
                        'odds1': float(o1),
                        'odds2': float(o2),
                        'timestamp': datetime.utcnow().isoformat(),
                        'source': source_name,
                    })
        except Exception:
            return []
        return results

    def _to_float(self, s: str) -> float:
        try:
            return float(s.replace(',', '.').strip())
        except Exception:
            return 0.0
