import asyncio
import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import json
import random
from tenacity import retry, stop_after_attempt, wait_exponential
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
    """Scraper ที่ยืดหยุ่นและทนทานต่อการเปลี่ยนแปลง"""

    def __init__(self):
        self.selectors_config = self._load_selectors_config()
        self.driver_pool = []
        self.session = None

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
            },
            OddsSource.GGBET: {
                "primary_selectors": [
                    "//div[@class='odd__value']",
                    "//button[contains(@class, 'odds-button')]"
                ],
                "dynamic_content": True,
                "scroll_required": True
            }
        }

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
