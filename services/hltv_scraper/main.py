"""
HLTV Scraper Service - Main Application
Production-ready scraper with anti-detection, proxy rotation, and monitoring
"""
import asyncio
import json
import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import hashlib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, Browser, BrowserContext
from bs4 import BeautifulSoup
import redis.asyncio as redis
from aiokafka import AIOKafkaProducer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
from config import Settings
from models import Match
from scrapers import HLTVMatchScraper, HLTVTeamScraper, HLTVPlayerScraper
from proxy_rotation import ProxyRotator
from anti_detection import AntiDetectionMiddleware
from validators import DataValidator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# Metrics
scrape_counter = Counter('hltv_scrapes_total', 'Total number of scrapes', ['type', 'status'])
scrape_duration = Histogram('hltv_scrape_duration_seconds', 'Scrape duration', ['type'])
active_browsers = Gauge('hltv_active_browsers', 'Number of active browser instances')
proxy_health = Gauge('hltv_proxy_health', 'Proxy health score')

class ScraperManager:
    """Manages browser instances and scraping operations"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.browser_pool: List[Browser] = []
        self.context_pool: List[BrowserContext] = []
        self.proxy_rotator = ProxyRotator(settings)
        self.anti_detection = AntiDetectionMiddleware()
        self.validator = DataValidator()
        self.redis_client = None
        self.kafka_producer = None
        self.db_engine = None

    async def initialize(self):
        """Initialize all connections and browser pool"""
        logger.info("Initializing scraper manager")

        # Redis connection
        self.redis_client = await redis.from_url(
            self.settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

        # Kafka producer
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.settings.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        )
        await self.kafka_producer.start()

        # Database engine
        self.db_engine = create_async_engine(
            self.settings.database_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )

        self.SessionLocal = sessionmaker(
            self.db_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Initialize browser pool
        await self._init_browser_pool()

        logger.info("Scraper manager initialized successfully")

    async def _init_browser_pool(self):
        """Initialize pool of browser instances"""
        playwright = await async_playwright().start()

        for _ in range(self.settings.browser_pool_size):
            proxy = await self.proxy_rotator.get_proxy()

            browser_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu',
                '--disable-blink-features=AutomationControlled',
                f"--user-agent={self.anti_detection.get_random_user_agent()}",
            ]

            browser = await playwright.chromium.launch(
                headless=self.settings.headless_mode,
                args=browser_args,
                proxy={
                    "server": f"http://{proxy['host']}:{proxy['port']}",
                    "username": proxy.get('username'),
                    "password": proxy.get('password'),
                } if proxy else None,
            )

            self.browser_pool.append(browser)
            active_browsers.inc()

            # Create contexts with anti-detection
            context = await self._create_stealth_context(browser)
            self.context_pool.append(context)

        logger.info(f"Browser pool initialized with {len(self.browser_pool)} instances")

    async def _create_stealth_context(self, browser: Browser) -> BrowserContext:
        """Create browser context with anti-detection measures"""
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York',
            permissions=['geolocation'],
            geolocation={'latitude': 40.7128, 'longitude': -74.0060},
            color_scheme='light',
            device_scale_factor=1,
            has_touch=False,
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
            },
        )

        # Inject anti-detection scripts
        await context.add_init_script(self.anti_detection.get_stealth_script())

        return context

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def scrape_matches(self, date_range: int = 7) -> List[Dict]:
        """Scrape upcoming matches"""
        start_time = time.time()
        matches: List[Dict] = []

        try:
            context = random.choice(self.context_pool)
            page = await context.new_page()

            # Random delay to appear human
            await asyncio.sleep(random.uniform(1, 3))

            # Navigate to matches page
            await page.goto(
                "https://www.hltv.org/matches",
                wait_until='networkidle',
                timeout=30000,
            )

            # Wait for content to load
            await page.wait_for_selector('.upcomingMatch', timeout=10000)

            # Extract match data
            html = await page.content()
            soup = BeautifulSoup(html, 'lxml')

            match_elements = soup.select('.upcomingMatch')[:50]  # Limit to 50 matches

            for element in match_elements:
                try:
                    match_data = self._extract_match_data(element)
                    if match_data and self.validator.validate_match(match_data):
                        matches.append(match_data)

                        # Store in cache
                        cache_key = f"match:{match_data['match_id']}"
                        await self.redis_client.setex(
                            cache_key,
                            3600,
                            json.dumps(match_data),
                        )

                        # Send to Kafka
                        await self.kafka_producer.send(
                            'hltv.matches',
                            value=match_data,
                        )

                except Exception as e:
                    logger.error(f"Error extracting match: {e}")
                    continue

            await page.close()

            # Record metrics
            duration = time.time() - start_time
            scrape_duration.labels(type='matches').observe(duration)
            scrape_counter.labels(type='matches', status='success').inc()

            logger.info(f"Scraped {len(matches)} matches in {duration:.2f}s")

            # Store in database
            await self._store_matches(matches)

            return matches

        except Exception as e:
            scrape_counter.labels(type='matches', status='error').inc()
            logger.error(f"Error scraping matches: {e}")
            raise

    def _extract_match_data(self, element) -> Optional[Dict]:
        """Extract match data from HTML element"""
        try:
            match_id = element.get('data-zonedgrouping-entry-unix')
            if not match_id:
                return None

            team1_elem = element.select_one('.team1 .matchTeamName') or element.select_one('.team1 .teamName')
            team2_elem = element.select_one('.team2 .matchTeamName') or element.select_one('.team2 .teamName')

            if not team1_elem or not team2_elem:
                return None

            match_time_elem = element.select_one('.matchTime')
            event_elem = element.select_one('.matchEventName')

            return {
                'match_id': hashlib.md5(str(match_id).encode()).hexdigest()[:16],
                'team1': team1_elem.text.strip(),
                'team2': team2_elem.text.strip(),
                'scheduled_time': match_time_elem.text.strip() if match_time_elem else None,
                'event_name': event_elem.text.strip() if event_elem else None,
                'format': element.select_one('.matchMeta').text.strip() if element.select_one('.matchMeta') else 'BO1',
                'scraped_at': datetime.utcnow().isoformat(),
                'source': 'hltv',
            }

        except Exception as e:
            logger.error(f"Error extracting match data: {e}")
            return None

    async def scrape_team_details(self, team_id: str) -> Optional[Dict]:
        """Scrape detailed team information"""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"team:{team_id}"
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)

            context = random.choice(self.context_pool)
            page = await context.new_page()

            # Random delay
            await asyncio.sleep(random.uniform(1, 3))

            # Navigate to team page
            await page.goto(
                f"https://www.hltv.org/team/{team_id}",
                wait_until='networkidle',
                timeout=30000,
            )

            # Wait for content
            await page.wait_for_selector('.teamProfile', timeout=10000)

            # Extract team data
            html = await page.content()
            soup = BeautifulSoup(html, 'lxml')

            team_data = {
                'team_id': team_id,
                'name': soup.select_one('.teamName').text.strip() if soup.select_one('.teamName') else None,
                'country': (soup.select_one('.team-country img') or {}).get('title') if soup.select_one('.team-country img') else None,
                'world_ranking': self._extract_ranking(soup),
                'players': self._extract_players(soup),
                'recent_results': self._extract_recent_results(soup),
                'map_stats': self._extract_map_stats(soup),
                'scraped_at': datetime.utcnow().isoformat(),
            }

            await page.close()

            if self.validator.validate_team(team_data):
                # Cache the data
                await self.redis_client.setex(cache_key, 7200, json.dumps(team_data))

                # Send to Kafka
                await self.kafka_producer.send('hltv.teams', value=team_data)

                # Record metrics
                duration = time.time() - start_time
                scrape_duration.labels(type='team').observe(duration)
                scrape_counter.labels(type='team', status='success').inc()

                return team_data

        except Exception as e:
            scrape_counter.labels(type='team', status='error').inc()
            logger.error(f"Error scraping team {team_id}: {e}")
            return None

    def _extract_ranking(self, soup) -> Optional[int]:
        """Extract team ranking"""
        try:
            ranking_elem = soup.select_one('.profile-team-stat:has(.ranking) .right')
            if ranking_elem:
                ranking_text = ranking_elem.text.strip()
                try:
                    return int(ranking_text.replace('#', '').strip())
                except ValueError:
                    return None
        except Exception:
            return None

    def _extract_players(self, soup) -> List[Dict]:
        """Extract team players"""
        players: List[Dict] = []
        try:
            player_elements = soup.select('.bodyshot-team .col-custom')
            for elem in player_elements:
                player_link = elem.select_one('a')
                if player_link and player_link.get('href'):
                    href = player_link['href'].strip('/').split('/')
                    pid = href[-2] if len(href) >= 2 else None
                    players.append({
                        'player_id': pid,
                        'nickname': (elem.select_one('.playerNickname').text.strip() if elem.select_one('.playerNickname') else None),
                        'real_name': (elem.select_one('.playerRealname').text.strip() if elem.select_one('.playerRealname') else None),
                    })
        except Exception as e:
            logger.error(f"Error extracting players: {e}")
        return players

    def _extract_recent_results(self, soup) -> List[Dict]:
        """Extract recent match results"""
        results: List[Dict] = []
        try:
            result_elements = soup.select('.result-box')[:10]
            for elem in result_elements:
                results.append({
                    'opponent': elem.select_one('.team').text.strip() if elem.select_one('.team') else None,
                    'score': elem.select_one('.result-score').text.strip() if elem.select_one('.result-score') else None,
                    'event': elem.select_one('.event-name').text.strip() if elem.select_one('.event-name') else None,
                })
        except Exception as e:
            logger.error(f"Error extracting results: {e}")
        return results

    def _extract_map_stats(self, soup) -> Dict:
        """Extract map statistics"""
        map_stats: Dict[str, Optional[str]] = {}
        try:
            map_elements = soup.select('.map-statistics .map-stat')
            for elem in map_elements:
                map_name = elem.select_one('.map-name').text.strip() if elem.select_one('.map-name') else None
                win_rate = elem.select_one('.map-winrate').text.strip() if elem.select_one('.map-winrate') else None
                if map_name:
                    map_stats[map_name] = win_rate
        except Exception as e:
            logger.error(f"Error extracting map stats: {e}")
        return map_stats

    async def scrape_live_scores(self) -> List[Dict]:
        """Scrape live match scores"""
        try:
            context = random.choice(self.context_pool)
            page = await context.new_page()

            await page.goto(
                "https://www.hltv.org/matches",
                wait_until='networkidle',
                timeout=30000,
            )

            # Wait for live matches
            await page.wait_for_selector('.liveMatch', timeout=5000)

            html = await page.content()
            soup = BeautifulSoup(html, 'lxml')

            live_matches: List[Dict] = []
            live_elements = soup.select('.liveMatch')

            for elem in live_elements:
                try:
                    match_data = {
                        'match_id': elem.get('data-match-id'),
                        'team1': elem.select_one('.team1 .teamName').text.strip() if elem.select_one('.team1 .teamName') else None,
                        'team2': elem.select_one('.team2 .teamName').text.strip() if elem.select_one('.team2 .teamName') else None,
                        'score1': elem.select_one('.team1 .currentMapScore').text.strip() if elem.select_one('.team1 .currentMapScore') else None,
                        'score2': elem.select_one('.team2 .currentMapScore').text.strip() if elem.select_one('.team2 .currentMapScore') else None,
                        'current_map': elem.select_one('.mapText').text.strip() if elem.select_one('.mapText') else None,
                        'timestamp': datetime.utcnow().isoformat(),
                    }

                    if self.validator.validate_live_score(match_data):
                        live_matches.append(match_data)

                        # Send to real-time stream
                        await self.kafka_producer.send('hltv.live', value=match_data)

                except Exception as e:
                    logger.error(f"Error extracting live match: {e}")
                    continue

            await page.close()

            scrape_counter.labels(type='live', status='success').inc()
            return live_matches

        except Exception as e:
            scrape_counter.labels(type='live', status='error').inc()
            logger.error(f"Error scraping live scores: {e}")
            return []

    async def _store_matches(self, matches: List[Dict]):
        """Store matches in database"""
        async with self.SessionLocal() as session:
            try:
                for match_data in matches:
                    # Check if match exists
                    existing = await session.execute(
                        select(Match).where(Match.match_id == match_data['match_id'])
                    )
                    if not existing.scalar():
                        match = Match(
                            match_id=match_data['match_id'],
                            team1_name=match_data['team1'],
                            team2_name=match_data['team2'],
                            event_name=match_data.get('event_name'),
                            format=match_data.get('format'),
                        )
                        session.add(match)

                await session.commit()
                logger.info(f"Stored {len(matches)} matches in database")

            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing matches: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up scraper resources")

        # Close browser contexts and instances
        for context in self.context_pool:
            await context.close()

        for browser in self.browser_pool:
            await browser.close()
            active_browsers.dec()

        # Close connections
        if self.kafka_producer:
            await self.kafka_producer.stop()

        if self.redis_client:
            await self.redis_client.close()

        if self.db_engine:
            await self.db_engine.dispose()

        logger.info("Cleanup completed")

# FastAPI Application
from fastapi import FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    settings = Settings()
    app.state.scraper = ScraperManager(settings)
    await app.state.scraper.initialize()

    # Start background tasks
    asyncio.create_task(periodic_scraping(app.state.scraper))

    yield

    # Shutdown
    await app.state.scraper.cleanup()

app = FastAPI(
    title="HLTV Scraper Service",
    version="1.0.0",
    description="Production-ready HLTV data scraping service",
    lifespan=lifespan,
)

# API Endpoints
class ScrapeRequest(BaseModel):
    target: str = Field(..., description="Target to scrape: matches, teams, players, live")
    options: Dict[str, Any] = Field(default_factory=dict)

class ScrapeResponse(BaseModel):
    status: str
    data: Optional[List[Dict]]
    count: int
    duration: float

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_browsers": active_browsers._value.get(),
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")

@app.post("/scrape", response_model=ScrapeResponse)
async def trigger_scrape(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Trigger scraping operation"""
    start_time = time.time()
    scraper = app.state.scraper

    try:
        if request.target == "matches":
            data = await scraper.scrape_matches()
        elif request.target == "teams":
            team_id = request.options.get('team_id')
            if not team_id:
                raise HTTPException(400, "team_id required for team scraping")
            team = await scraper.scrape_team_details(team_id)
            data = [team] if team else []
        elif request.target == "live":
            data = await scraper.scrape_live_scores()
        else:
            raise HTTPException(400, f"Invalid target: {request.target}")

        duration = time.time() - start_time

        return ScrapeResponse(
            status="success",
            data=data,
            count=len(data),
            duration=duration,
        )

    except Exception as e:
        logger.error(f"Scrape error: {e}")
        raise HTTPException(500, str(e))

@app.get("/jobs")
async def list_jobs():
    """List scraping jobs"""
    scraper = app.state.scraper

    # Get jobs from Redis
    jobs = []
    keys = await scraper.redis_client.keys("job:*")

    for key in keys:
        job_data = await scraper.redis_client.get(key)
        if job_data:
            jobs.append(json.loads(job_data))

    return {"jobs": jobs, "count": len(jobs)}

async def periodic_scraping(scraper: ScraperManager):
    """Background task for periodic scraping"""
    while True:
        try:
            # Scrape matches every 5 minutes
            await scraper.scrape_matches()
            await asyncio.sleep(300)

            # Scrape live scores every 30 seconds
            await scraper.scrape_live_scores()
            await asyncio.sleep(30)

        except Exception as e:
            logger.error(f"Periodic scraping error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
