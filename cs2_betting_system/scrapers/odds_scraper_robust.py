import asyncio
# aiohttp import with fallback handling
try:
    import aiohttp
except ImportError:
    aiohttp = None
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import json
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddsSource(Enum):
    ODDSPORTAL = "oddsportal"
    BET365 = "bet365"
    PINNACLE = "pinnacle"
    GGBET = "ggbet"
    ONEXBET = "1xbet"
    THUNDERPICK = "thunderpick"
    BETWAY = "betway"
    RIVALRY = "rivalry"


@dataclass
class OddsRecord:
    """ข้อมูลราคาพร้อม metadata"""
    team1: str
    team2: str
    odds_1: float
    odds_2: float
    odds_draw: Optional[float]
    source: str
    timestamp: datetime
    match_time: Optional[datetime]
    market_type: str  # match_winner, map1_winner, etc.
    confidence: float  # 0-1 ความมั่นใจในข้อมูล
    raw_data: Optional[Dict]  # เก็บข้อมูลดิบ

    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.match_time:
            data['match_time'] = self.match_time.isoformat()
        return data


class FlexibleOddsScraper:
    """Enhanced Robust Scraper with retry/backoff and source tracking"""

    def __init__(self):
        self.selectors_config = self._load_selectors_config()
        self.driver_pool = []
        self.session = None
        self.source_reliability = {}  # Track source success rates
        self.last_attempt_times = {}  # Rate limiting per source
        self.failed_attempts = {}  # Track consecutive failures

    def _load_selectors_config(self) -> Dict:
        """โหลด selector config ที่อัปเดตได้ง่าย"""
        return {
            OddsSource.ODDSPORTAL: {
                "primary_selectors": [
                    "//div[@class='odds-cell__value']",
                    "//span[@class='odds-value']",
                    "//td[contains(@class, 'odds')]//a"
                ],
                "fallback_selectors": [
                    "//div[contains(@class, 'odd')]",
                    "//span[contains(text(), '.')][@class]",
                    "//td[@xodd]"
                ],
                "team_selectors": [
                    "//div[@class='event-team']",
                    "//span[@class='team-name']",
                    "//a[contains(@class, 'participant')]"
                ],
                "wait_conditions": [
                    (By.CLASS_NAME, "odds-cell__value"),
                    (By.XPATH, "//div[contains(@class, 'odds')]")
                ]
            },
            OddsSource.BET365: {
                "primary_selectors": [
                    "//span[contains(@class, 'sgl-ParticipantOddsOnly')]",
                    "//div[@class='cm-MarketOdds']//span"
                ],
                "api_endpoint": "https://api.bet365.com/odds",
                "requires_auth": True
            },
            OddsSource.PINNACLE: {
                "api_endpoint": "https://api.pinnacle.com/v3/odds",
                "headers": {
                    "Accept": "application/json",
                    "X-API-Key": "YOUR_API_KEY"
                }
            }
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def fetch_odds_with_retry(self, source: OddsSource, match_identifiers: Dict) -> Optional[List[OddsRecord]]:
        """Fetch odds with exponential backoff retry"""
        source_name = source.value
        
        # Rate limiting check
        if source_name in self.last_attempt_times:
            time_since_last = time.time() - self.last_attempt_times[source_name]
            min_interval = 2.0  # Minimum 2 seconds between requests
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
        
        self.last_attempt_times[source_name] = time.time()
        
        try:
            # Track attempt
            if source_name not in self.failed_attempts:
                self.failed_attempts[source_name] = 0
                
            # Use different methods based on source config
            config = self.selectors_config.get(source, {})
            
            if "api_endpoint" in config:
                odds = await self._fetch_via_api(source, match_identifiers, config)
            else:
                odds = await self._fetch_via_scraping(source, match_identifiers, config)
            
            # Success - reset failure counter and update reliability
            self.failed_attempts[source_name] = 0
            self._update_source_reliability(source_name, True)
            
            return odds
            
        except Exception as e:
            # Track failure
            self.failed_attempts[source_name] += 1
            self._update_source_reliability(source_name, False)
            
            logger.warning(f"Failed to fetch odds from {source_name}: {e}")
            
            # If too many consecutive failures, temporarily disable source
            if self.failed_attempts[source_name] >= 5:
                logger.error(f"Source {source_name} disabled due to consecutive failures")
                return None
                
            raise  # Re-raise for retry mechanism
    
    def _update_source_reliability(self, source_name: str, success: bool):
        """Update source reliability tracking"""
        if source_name not in self.source_reliability:
            self.source_reliability[source_name] = {"successes": 0, "attempts": 0}
        
        self.source_reliability[source_name]["attempts"] += 1
        if success:
            self.source_reliability[source_name]["successes"] += 1
    
    def get_source_reliability_score(self, source_name: str) -> float:
        """Get reliability score for a source (0.0 to 1.0)"""
        if source_name not in self.source_reliability:
            return 0.5  # Default neutral score
        
        stats = self.source_reliability[source_name]
        if stats["attempts"] == 0:
            return 0.5
        
        return stats["successes"] / stats["attempts"]
    
    async def _fetch_via_api(self, source: OddsSource, match_identifiers: Dict, config: Dict) -> List[OddsRecord]:
        """Fetch odds via API with proper error handling"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        endpoint = config["api_endpoint"]
        headers = config.get("headers", {})
        
        # Build API request based on source
        if source == OddsSource.PINNACLE:
            params = {
                "sportId": 12,  # Esports
                "leagueIds": match_identifiers.get("league_id", ""),
                "eventIds": match_identifiers.get("event_id", "")
            }
        else:
            params = match_identifiers
        
        async with self.session.get(endpoint, headers=headers, params=params, timeout=10) as response:
            if response.status != 200:
                raise ConnectionError(f"API returned status {response.status}")
            
            data = await response.json()
            return self._parse_api_response(source, data)
    
    async def _fetch_via_scraping(self, source: OddsSource, match_identifiers: Dict, config: Dict) -> List[OddsRecord]:
        """Fetch odds via web scraping with fallback selectors"""
        driver = await self._get_driver()
        
        try:
            # Navigate to odds page
            url = self._build_odds_url(source, match_identifiers)
            driver.get(url)
            
            # Wait for page load with multiple conditions
            wait_conditions = config.get("wait_conditions", [])
            for condition_type, condition_value in wait_conditions:
                try:
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((condition_type, condition_value))
                    )
                    break
                except:
                    continue
            
            # Try primary selectors first, then fallbacks
            odds_data = None
            
            for selector_type in ["primary_selectors", "fallback_selectors"]:
                selectors = config.get(selector_type, [])
                for selector in selectors:
                    try:
                        odds_data = self._extract_odds_with_selector(driver, selector, config)
                        if odds_data:
                            break
                    except Exception as e:
                        logger.debug(f"Selector {selector} failed: {e}")
                        continue
                
                if odds_data:
                    break
            
            if not odds_data:
                raise ValueError(f"No odds data found for {source.value}")
            
            return self._convert_to_odds_records(source, odds_data, match_identifiers)
            
        finally:
            await self._return_driver(driver)
    
    def _extract_odds_with_selector(self, driver, selector: str, config: Dict) -> Optional[Dict]:
        """Extract odds using specific selector with validation"""
        elements = driver.find_elements(By.XPATH, selector)
        
        if not elements:
            return None
        
        odds_values = []
        for element in elements:
            try:
                text = element.text.strip()
                if text and self._is_valid_odds(text):
                    odds_values.append(float(text))
            except (ValueError, AttributeError):
                continue
        
        # Validate we have expected number of odds (typically 2 for match winner)
        if len(odds_values) >= 2:
            return {
                "team1_odds": odds_values[0],
                "team2_odds": odds_values[1],
                "draw_odds": odds_values[2] if len(odds_values) > 2 else None
            }
        
        return None
    
    def _is_valid_odds(self, text: str) -> bool:
        """Validate if text represents valid odds"""
        try:
            odds = float(text)
            return 1.01 <= odds <= 100.0  # Reasonable odds range
        except ValueError:
            return False
    
    async def _get_driver(self):
        """Get WebDriver from pool or create new one"""
        if self.driver_pool:
            return self.driver_pool.pop()
        
        # Create new driver with anti-detection measures
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
    async def _return_driver(self, driver):
        """Return driver to pool for reuse"""
        if len(self.driver_pool) < 3:  # Max 3 drivers in pool
            self.driver_pool.append(driver)
        else:
            driver.quit()
    
    def _build_odds_url(self, source: OddsSource, match_identifiers: Dict) -> str:
        """Build URL for odds page based on source and match identifiers"""
        base_urls = {
            OddsSource.ODDSPORTAL: "https://www.oddsportal.com/esports/counter-strike",
            OddsSource.BET365: "https://www.bet365.com/esports/counter-strike",
            OddsSource.GGBET: "https://gg.bet/en/esports/counter-strike"
        }
        
        base_url = base_urls.get(source, "")
        match_path = match_identifiers.get("url_path", "")
        
        return f"{base_url}/{match_path}" if match_path else base_url
    
    def _parse_api_response(self, source: OddsSource, data: Dict) -> List[OddsRecord]:
        """Parse API response into OddsRecord objects"""
        records = []
        
        # Implementation depends on API structure
        # This is a template that should be customized per source
        
        return records
    
    def _convert_to_odds_records(self, source: OddsSource, odds_data: Dict, match_identifiers: Dict) -> List[OddsRecord]:
        """Convert extracted odds data to OddsRecord objects"""
        record = OddsRecord(
            team1=match_identifiers.get("team1", "Team1"),
            team2=match_identifiers.get("team2", "Team2"),
            odds_1=odds_data["team1_odds"],
            odds_2=odds_data["team2_odds"],
            odds_draw=odds_data.get("draw_odds"),
            source=source.value,
            timestamp=datetime.utcnow(),
            match_time=match_identifiers.get("match_time"),
            market_type="match_winner",
            confidence=self.get_source_reliability_score(source.value),
            raw_data=odds_data
        )
        
        return [record]

    async def setup(self):
        """Initialize resources"""
        self.session = aiohttp.ClientSession()
        # สร้าง driver pool สำหรับ parallel scraping
        for _ in range(3):
            driver = self._create_driver()
            self.driver_pool.append(driver)

    def _create_driver(self):
        """สร้าง selenium driver พร้อม anti-detection"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument(f'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_odds_with_retry(self, match_url: str, source: OddsSource) -> Optional[OddsRecord]:
        """ดึงราคาพร้อม retry mechanism"""
        try:
            if source in [OddsSource.PINNACLE, OddsSource.BET365]:
                return await self._fetch_api_odds(match_url, source)
            else:
                return await self._fetch_scrape_odds(match_url, source)
        except Exception as e:
            logger.warning(f"Failed to fetch from {source.value}: {e}")
            raise

    async def _fetch_scrape_odds(self, match_url: str, source: OddsSource) -> Optional[OddsRecord]:
        """ดึงราคาผ่าน web scraping"""
        driver = self._get_available_driver()
        config = self.selectors_config.get(source, {})

        try:
            driver.get(match_url)

            # รอให้หน้าโหลด
            wait = WebDriverWait(driver, 10)
            for condition in config.get("wait_conditions", []):
                try:
                    wait.until(EC.presence_of_element_located(condition))
                    break
                except:
                    continue

            # ลองหา odds ด้วย selector หลายตัว
            odds = self._extract_odds_flexible(driver, config)
            teams = self._extract_teams_flexible(driver, config)

            if odds and teams and len(odds) >= 2:
                return OddsRecord(
                    team1=teams[0],
                    team2=teams[1],
                    odds_1=odds[0],
                    odds_2=odds[1],
                    odds_draw=odds[2] if len(odds) > 2 else None,
                    source=source.value,
                    timestamp=datetime.now(),
                    match_time=self._extract_match_time(driver),
                    market_type="match_winner",
                    confidence=self._calculate_confidence(odds, teams),
                    raw_data={"url": match_url, "odds": odds, "teams": teams}
                )
        finally:
            self._return_driver(driver)

        return None

    def _extract_odds_flexible(self, driver, config: Dict) -> List[float]:
        """ดึง odds แบบยืดหยุ่น"""
        odds = []

        # ลอง primary selectors ก่อน
        for selector in config.get("primary_selectors", []):
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    for elem in elements:
                        try:
                            text = elem.text.strip()
                            # ทำความสะอาดข้อมูล
                            value = float(text.replace(',', '.').replace('+', ''))
                            if 1.0 <= value <= 100.0:  # ตรวจสอบช่วงที่เป็นไปได้
                                odds.append(value)
                        except:
                            continue
                    if odds:
                        return odds[:3]  # เอาแค่ 3 ตัวแรก
            except:
                continue

        # ถ้าไม่เจอ ลอง fallback selectors
        for selector in config.get("fallback_selectors", []):
            try:
                elements = driver.find_elements(By.XPATH, selector)
                for elem in elements:
                    try:
                        text = elem.text.strip()
                        if '.' in text and len(text) <= 6:
                            value = float(text)
                            if 1.0 <= value <= 100.0:
                                odds.append(value)
                    except:
                        continue
                if len(odds) >= 2:
                    return odds[:3]
            except:
                continue

        return odds

    def _extract_teams_flexible(self, driver, config: Dict) -> List[str]:
        """ดึงชื่อทีมแบบยืดหยุ่น"""
        teams = []

        for selector in config.get("team_selectors", []):
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    for elem in elements:
                        text = elem.text.strip()
                        if text and len(text) > 1:
                            teams.append(text)
                    if len(teams) >= 2:
                        return teams[:2]
            except:
                continue

        return teams

    async def _fetch_api_odds(self, match_id: str, source: OddsSource) -> Optional[OddsRecord]:
        """ดึงราคาผ่าน API"""
        config = self.selectors_config.get(source, {})
        endpoint = config.get("api_endpoint")
        headers = config.get("headers", {})

        if not endpoint:
            return None

        try:
            async with self.session.get(
                f"{endpoint}/{match_id}",
                headers=headers,
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_api_response(data, source)
        except Exception as e:
            logger.error(f"API fetch failed for {source.value}: {e}")

        return None

    def _calculate_confidence(self, odds: List[float], teams: List[str]) -> float:
        """คำนวณความมั่นใจในข้อมูล"""
        confidence = 1.0

        # ตรวจสอบความสมบูรณ์
        if not odds or not teams:
            return 0.0

        if len(odds) < 2:
            confidence *= 0.5

        if len(teams) < 2:
            confidence *= 0.5

        # ตรวจสอบความสมเหตุสมผลของ odds
        if any(o < 1.0 or o > 100.0 for o in odds):
            confidence *= 0.7

        # ตรวจสอบ margin
        if len(odds) >= 2:
            margin = (1/odds[0] + 1/odds[1]) - 1
            if margin < -0.1 or margin > 0.3:  # margin ผิดปกติ
                confidence *= 0.8

        return confidence

    def _get_available_driver(self):
        """ดึง driver ที่ว่าง"""
        # Implement driver pool management
        return self.driver_pool[0]

    def _return_driver(self, driver):
        """คืน driver กลับ pool"""
        pass

    def _extract_match_time(self, driver) -> Optional[datetime]:
        """ดึงเวลาแข่ง"""
        # Implementation specific to each site
        return None


class MultiSourceOddsFetcher:
    """ระบบดึงราคาจากหลายแหล่ง"""

    def __init__(self):
        self.scrapers = {}
        self.priority_sources = [
            OddsSource.PINNACLE,
            OddsSource.BET365,
            OddsSource.ODDSPORTAL,
            OddsSource.GGBET,
            OddsSource.ONEXBET
        ]

    async def initialize(self):
        """Setup all scrapers"""
        for source in self.priority_sources:
            scraper = FlexibleOddsScraper()
            await scraper.setup()
            self.scrapers[source] = scraper

    async def fetch_odds_with_fallback(
        self,
        match_identifier: Dict[str, str],
        required_confidence: float = 0.7
    ) -> List[OddsRecord]:
        """ดึงราคาจากหลายแหล่งพร้อม fallback"""

        odds_records = []
        tasks = []

        # สร้าง tasks สำหรับทุก source
        for source in self.priority_sources:
            if source in match_identifier:
                task = self._fetch_from_source(
                    match_identifier[source],
                    source
                )
                tasks.append(task)

        # รอผลลัพธ์
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # กรองผลลัพธ์ที่ใช้ได้
        for result in results:
            if isinstance(result, OddsRecord):
                if result.confidence >= required_confidence:
                    odds_records.append(result)

        # ถ้าไม่มีผลลัพธ์ที่ดีพอ ลองแหล่งสำรอง
        if len(odds_records) < 2:
            backup_sources = [
                OddsSource.THUNDERPICK,
                OddsSource.BETWAY,
                OddsSource.RIVALRY
            ]

            for source in backup_sources:
                if source in match_identifier:
                    try:
                        result = await self._fetch_from_source(
                            match_identifier[source],
                            source
                        )
                        if result and result.confidence >= 0.5:
                            odds_records.append(result)
                    except:
                        continue

        return odds_records

    async def _fetch_from_source(
        self,
        match_url: str,
        source: OddsSource
    ) -> Optional[OddsRecord]:
        """ดึงจากแหล่งเดียว"""
        scraper = self.scrapers.get(source)
        if not scraper:
            return None

        try:
            return await scraper.fetch_odds_with_retry(match_url, source)
        except Exception as e:
            logger.error(f"Failed to fetch from {source.value}: {e}")
            return None

    def aggregate_odds(self, odds_records: List[OddsRecord]) -> Dict:
        """รวมราคาจากหลายแหล่ง"""
        if not odds_records:
            return None

        # Group by teams
        team_odds = {}
        for record in odds_records:
            key = f"{record.team1}_vs_{record.team2}"
            if key not in team_odds:
                team_odds[key] = {
                    "team1": record.team1,
                    "team2": record.team2,
                    "odds_1": [],
                    "odds_2": [],
                    "odds_draw": [],
                    "sources": [],
                    "timestamps": []
                }

            team_odds[key]["odds_1"].append((record.odds_1, record.confidence))
            team_odds[key]["odds_2"].append((record.odds_2, record.confidence))
            if record.odds_draw:
                team_odds[key]["odds_draw"].append((record.odds_draw, record.confidence))
            team_odds[key]["sources"].append(record.source)
            team_odds[key]["timestamps"].append(record.timestamp)

        # Calculate weighted average
        aggregated = {}
        for key, data in team_odds.items():
            aggregated[key] = {
                "team1": data["team1"],
                "team2": data["team2"],
                "odds_1": self._weighted_average(data["odds_1"]),
                "odds_2": self._weighted_average(data["odds_2"]),
                "odds_draw": self._weighted_average(data["odds_draw"]) if data["odds_draw"] else None,
                "best_odds_1": max(o[0] for o in data["odds_1"]),
                "best_odds_2": max(o[0] for o in data["odds_2"]),
                "sources": data["sources"],
                "num_sources": len(data["sources"]),
                "last_update": max(data["timestamps"]),
                "confidence": sum(o[1] for o in data["odds_1"]) / len(data["odds_1"])
            }

        return aggregated

    def _weighted_average(self, odds_confidence_pairs: List[Tuple[float, float]]) -> float:
        """คำนวณค่าเฉลี่ยถ่วงน้ำหนักตาม confidence"""
        if not odds_confidence_pairs:
            return 0.0

        total_weight = sum(conf for _, conf in odds_confidence_pairs)
        if total_weight == 0:
            return sum(odds for odds, _ in odds_confidence_pairs) / len(odds_confidence_pairs)

        weighted_sum = sum(odds * conf for odds, conf in odds_confidence_pairs)
        return weighted_sum / total_weight
