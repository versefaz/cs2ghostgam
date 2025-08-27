#!/usr/bin/env python3
"""
Robust Multi-Source Odds Scraper
Fetches odds from multiple bookmakers with fallback, validation, and retry logic
"""

import asyncio
import json
import logging
import random
import aiohttp
import backoff
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import statistics
from enum import Enum

# Import SessionManager for proper session handling
from app.scrapers.session_manager import session_manager

# Import timing utilities and config
from core.utils.timing import HumanLikeTiming, TimingConfig, RateLimitError, RequestThrottler, random_startup_delay
from app.config import SCRAPER_SETTINGS, USER_AGENTS, RATE_LIMIT_CODES, SUCCESS_CODES

logger = logging.getLogger(__name__)


class BookmakerSource(Enum):
    ODDSPORTAL = "oddsportal"
    BET365 = "bet365"
    PINNACLE = "pinnacle"
    BETWAY = "betway"
    UNIBET = "unibet"
    GGBET = "ggbet"
    RIVALRY = "rivalry"


@dataclass
class OddsData:
    """Standardized odds data structure"""
    match_id: str
    bookmaker: str
    market: str  # match_winner, handicap, total_maps, etc.
    odds: Dict[str, float]  # side -> odds
    timestamp: datetime
    source_url: str
    confidence: float = 1.0  # Confidence in data accuracy
    
    # Metadata
    last_updated: Optional[datetime] = None
    movement: Optional[Dict[str, float]] = None  # Previous odds for comparison
    volume: Optional[int] = None  # Betting volume if available


@dataclass
class MarketConsensus:
    """Consensus odds across multiple bookmakers"""
    match_id: str
    market: str
    consensus_odds: Dict[str, float]  # Average odds
    best_odds: Dict[str, Tuple[float, str]]  # Best odds and bookmaker
    spread: Dict[str, float]  # Odds spread (max - min)
    bookmaker_count: int
    sources: List[str]
    timestamp: datetime
    
    # Arbitrage detection
    arbitrage_opportunity: bool = False
    arbitrage_profit: float = 0.0
    arbitrage_stakes: Optional[Dict[str, float]] = None


class RobustOddsScraper:
    """Robust odds scraper with multi-source support, validation, and consensus"""
    
    def __init__(self):
        # Remove direct session management - use SessionManager instead
        self.timing = HumanLikeTiming(TimingConfig(
            base_interval_sec=3.0,
            jitter_pct=0.3,
            max_backoff=5,
            min_delay=1.0
        ))
        
        # Initialize throttlers for each bookmaker
        self.throttlers: Dict[str, RequestThrottler] = {}
        for source in BookmakerSource:
            self.throttlers[source.value] = RequestThrottler(
                max_concurrent=SCRAPER_SETTINGS['concurrency']['max_connections'],
                min_delay=SCRAPER_SETTINGS['rate_limiting']['min_delay_between_requests']
            )
        
        # Enhanced user agent rotation with more realistic agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        self.current_ua_index = 0
        self.timeout = 30
        
        # Enhanced bookmaker configurations with API endpoints
        self.bookmaker_configs = self._init_enhanced_bookmaker_configs()
        
        # Caching and validation
        self.odds_cache: Dict[str, List[OddsData]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.validation_threshold = 0.1  # 10% variance allowed
        
        # Rate limiting tracking
        self.rate_limits: Dict[str, Dict] = {}
        
        # Consensus settings
        self.min_sources_for_consensus = 2
        self.outlier_threshold = 2.0  # Standard deviations
        
        # Bookmaker-specific delays (in addition to global timing)
        self.bookmaker_delays: Dict[str, float] = {
            'gg.bet': 3.0,
            'rivalry': 2.5,
            'oddsportal': 4.0,
            'bet365': 5.0,
            'pinnacle': 3.5,
            'betway': 2.8,
            'unibet': 3.2
        }
    
    def _init_enhanced_bookmaker_configs(self) -> Dict[BookmakerSource, Dict]:
        """Initialize enhanced bookmaker configurations with API endpoints"""
        return {
            BookmakerSource.RIVALRY: {
                'base_url': 'https://www.rivalry.com',
                'api_endpoint': '/api/v4/matches/csgo',
                'web_endpoint': '/esports/cs2-betting',
                'rate_limit': 30,
                'request_type': 'api',  # 'api' or 'html'
                'headers': {
                    'Referer': 'https://www.rivalry.com/esports/cs2-betting',
                    'Accept': 'application/json, text/plain, */*',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                'accept': 'application/json'
            },
            BookmakerSource.GGBET: {
                'base_url': 'https://gg.bet',
                'web_endpoint': '/en/esports/betting/counter-strike',
                'rate_limit': 20,
                'request_type': 'html',
                'selector': 'div.event-row, .match-item, .betting-event',
                'headers': {
                    'Referer': 'https://gg.bet/en/esports',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                }
            },
            BookmakerSource.ODDSPORTAL: {
                'base_url': 'https://www.oddsportal.com',
                'web_endpoint': '/esports/counter-strike',
                'rate_limit': 15,
                'request_type': 'html',
                'selector': '.table-main tr, .event-row',
                'headers': {
                    'Referer': 'https://www.oddsportal.com/esports',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                }
            }
        }
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _get_next_user_agent(self) -> str:
        """Get next user agent from rotation"""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua
    
    async def initialize(self):
        """Initialize scraper - sessions managed by SessionManager"""
        logger.info(f"Initializing RobustOddsScraper with {len(self.bookmaker_configs)} bookmaker sources")
        
        # Pre-create sessions for all bookmakers
        for source in BookmakerSource:
            config = self.bookmaker_configs.get(source, {})
            headers = {
                'User-Agent': self._get_next_user_agent(),
                'Accept': config.get('accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                **config.get('headers', {})
            }
            
            # Get session from SessionManager
            await session_manager.get_session(
                source.value,
                headers=headers,
                timeout=self.timeout,
                limit=3,
                limit_per_host=2
            )
        
        logger.info(f"Initialized {len(BookmakerSource)} bookmaker sessions")
        
        logger.info(f"Initialized {len(self.sessions)} bookmaker sessions")
    
    async def close(self):
        """Close all HTTP sessions via SessionManager"""
        await session_manager.close_all()
        logger.info("All odds scraper sessions closed via SessionManager")
        await asyncio.sleep(0.1)
    
    async def _rate_limit_check(self, source: BookmakerSource):
        """Check and enforce rate limits"""
        current_time = asyncio.get_event_loop().time()
        rate_info = self.rate_limits.get(source.value, {'requests': 0, 'window_start': 0, 'max_per_minute': 20})
        rate_info = self.rate_limits[source]
        
        # Reset window if needed
        if current_time - rate_info['window_start'] >= 60:
            rate_info['requests'] = 0
            rate_info['window_start'] = current_time
        
        # Check if we're at the limit
        if rate_info['requests'] >= rate_info['max_per_minute']:
            sleep_time = 60 - (current_time - rate_info['window_start'])
            if sleep_time > 0:
                logger.debug(f"Rate limiting {source.value}, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                rate_info['requests'] = 0
                rate_info['window_start'] = asyncio.get_event_loop().time()
        
        rate_info['requests'] += 1
        
        # Add random delay
        await asyncio.sleep(0.5 + (asyncio.get_event_loop().time() % 1.0))
    
    async def _make_request(self, url: str, source: BookmakerSource, retries: int = 3) -> Optional[str]:
        """Make HTTP request with enhanced anti-bot measures"""
        config = self.bookmaker_configs.get(source, {})
        
        for attempt in range(retries):
            try:
                # Anti-bot delay with randomization
                base_delay = random.uniform(2.0, 5.0)
                jitter = random.uniform(-0.5, 0.5)
                await asyncio.sleep(base_delay + jitter + (attempt * 1.0))
                
                # Get session from SessionManager
                session = await session_manager.get_session(source.value)
                
                # Rotate user agent for each request
                headers = {
                    'User-Agent': self._get_next_user_agent(),
                    **config.get('headers', {})
                }
                
                # Add random headers to appear more human-like
                if random.random() > 0.5:
                    headers['Cache-Control'] = 'no-cache'
                if random.random() > 0.7:
                    headers['Pragma'] = 'no-cache'
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        self.timing.on_success()
                        return content
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        delay = self.timing.on_rate_limit(retry_after)
                        logger.warning(f"{source.value} rate limited, waiting {delay}s")
                        await asyncio.sleep(delay)
                    elif response.status == 403:  # Forbidden - likely bot detection
                        logger.warning(f"{source.value} returned 403 - possible bot detection")
                        await asyncio.sleep(random.uniform(10, 20))  # Longer delay
                    else:
                        logger.warning(f"{source.value} returned status {response.status}")
                        
            except Exception as e:
                self.timing.on_error("network" if "timeout" in str(e).lower() else "general")
                logger.error(f"Error fetching {url} from {source.value} (attempt {attempt + 1}): {e}")
                
                if attempt < retries - 1:
                    await asyncio.sleep(random.uniform(5, 10))  # Backoff delay
        
        return None

    async def _execute_request(self, url: str, source: BookmakerSource) -> Optional[str]:
        """Execute the actual HTTP request"""
        session = self.sessions[source.value]
        
        request_headers = {
            'User-Agent': self._get_next_user_agent(),
            'Referer': self.bookmaker_configs[source]['base_url'],
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        async with session.get(url, headers=request_headers) as response:
            if response.status in SUCCESS_CODES:
                self.timing.on_success()
                return await response.text()
            elif response.status in RATE_LIMIT_CODES:
                retry_after = response.headers.get('Retry-After')
                retry_after_int = int(retry_after) if retry_after and retry_after.isdigit() else None
                raise RateLimitError(response.status, retry_after_int, f"Rate limited by {source.value}")
            elif response.status in [403, 404]:
                logger.warning(f"HTTP {response.status} for {url} from {source.value}")
                return None
            else:
                logger.warning(f"HTTP {response.status} for {url} from {source.value}")
                return None

    async def run_continuous_scraping(self):
        """Run continuous odds scraping with human-like timing"""
        logger.info("Starting continuous odds scraping with human-like patterns")
        
        while True:
            try:
                # Scrape odds from all sources
                all_odds = await self.get_comprehensive_odds()
                logger.info(f"Scraped odds for {len(all_odds)} matches from multiple bookmakers")
                
                # Reset backoff on success
                self.timing.on_success()
                
            except RateLimitError:
                logger.warning("Rate limited, applying exponential backoff")
                delay = self.timing.on_rate_limit()
                await asyncio.sleep(delay)
                continue
                
            except Exception as e:
                logger.error(f"Odds scraping error: {e}")
                self.timing.on_error()
            
            # Wait with jitter before next scraping cycle
            delay = self.timing.next_delay()
            logger.debug(f"Next odds scrape in {delay:.1f} seconds")
            await asyncio.sleep(delay)

    async def scrape_oddsportal(self, sport: str = "esports") -> List[OddsData]:
        """Scrape odds from OddsPortal"""
        odds_data = []
        
        try:
            url = f"https://www.oddsportal.com/{sport}/counter-strike/"
            response = await self._make_request(url, BookmakerSource.ODDSPORTAL)
            
            if not response:
                return odds_data
            
            soup = BeautifulSoup(response, 'html.parser')
            
            # Updated selectors for current OddsPortal layout
            match_rows = soup.select('div[data-v-69e0c3c6] .eventRow, .table-main tbody tr')
            
            for row in match_rows:
                try:
                    # Extract team names - try multiple selectors
                    team_elements = row.select('.participant-name, .name, .team-name')
                    if len(team_elements) < 2:
                        continue
                    
                    team1 = team_elements[0].get_text(strip=True)
                    team2 = team_elements[1].get_text(strip=True)
                    
                    if not team1 or not team2:
                        continue
                    
                    # Extract odds - try multiple selectors
                    odds_elements = row.select('.odds-nowrp, .odds, .odd')
                    
                    if len(odds_elements) >= 2:
                        try:
                            odds1 = float(odds_elements[0].get_text(strip=True))
                            odds2 = float(odds_elements[1].get_text(strip=True))
                            
                            # Validate odds
                            if not self._validate_odds_values([odds1, odds2]):
                                continue
                            
                            # Generate match ID
                            match_id = f"oddsportal_{hash(f'{team1}_{team2}')}"
                            
                            odds_data.append(OddsData(
                                match_id=match_id,
                                bookmaker="oddsportal",
                                market="match_winner",
                                odds={team1.lower(): odds1, team2.lower(): odds2},
                                timestamp=datetime.utcnow(),
                                source_url=url,
                                confidence=0.9
                            ))
                            
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Error parsing odds: {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error processing OddsPortal row: {e}")
                    continue
            
            logger.info(f"Scraped {len(odds_data)} matches from OddsPortal")
            
        except Exception as e:
            logger.error(f"Error scraping OddsPortal: {e}")
        
        return odds_data
    
    async def scrape_ggbet(self) -> List[OddsData]:
        """Scrape odds from GG.bet"""
        odds_data = []
        
        try:
            url = "https://gg.bet/en/counter-strike"
            response = await self._make_request(url, BookmakerSource.GGBET)
            
            if not response:
                return odds_data
            
            soup = BeautifulSoup(response, 'html.parser')
            
            # GG.bet specific selectors
            match_elements = soup.select('.match-item, .event-item, .game-item')
            
            for match in match_elements:
                try:
                    # Team names
                    teams = match.select('.team-name, .participant-name')
                    if len(teams) < 2:
                        continue
                    
                    team1 = teams[0].get_text(strip=True)
                    team2 = teams[1].get_text(strip=True)
                    
                    # Odds
                    odds_elements = match.select('.odds-value, .coefficient, .odd')
                    if len(odds_elements) >= 2:
                        odds1 = float(odds_elements[0].get_text(strip=True))
                        odds2 = float(odds_elements[1].get_text(strip=True))
                        
                        if self._validate_odds_values([odds1, odds2]):
                            match_id = f"ggbet_{hash(f'{team1}_{team2}')}"
                            
                            odds_data.append(OddsData(
                                match_id=match_id,
                                bookmaker="ggbet",
                                market="match_winner",
                                odds={team1.lower(): odds1, team2.lower(): odds2},
                                timestamp=datetime.utcnow(),
                                source_url=url,
                                confidence=0.85
                            ))
                
                except Exception as e:
                    logger.debug(f"Error processing GG.bet match: {e}")
                    continue
            
            logger.info(f"Scraped {len(odds_data)} matches from GG.bet")
            
        except Exception as e:
            logger.error(f"Error scraping GG.bet: {e}")
        
        return odds_data
    
    async def scrape_pinnacle_api(self) -> List[OddsData]:
        """Scrape odds from Pinnacle API (if available)"""
        odds_data = []
        
        try:
            # This would require Pinnacle API credentials
            # For now, return empty list - to be implemented with proper API access
            logger.debug("Pinnacle API scraping not implemented (requires credentials)")
            
        except Exception as e:
            logger.error(f"Error scraping Pinnacle API: {e}")
        
        return odds_data
    
    async def scrape_rivalry(self) -> List[OddsData]:
        """Scrape odds from Rivalry"""
        odds_data = []
        
        try:
            url = "https://www.rivalry.com/esports/counter-strike-bets"
            response = await self._make_request(url, BookmakerSource.RIVALRY)
            
            if not response:
                return odds_data
            
            soup = BeautifulSoup(response, 'html.parser')
            
            # Rivalry specific selectors
            matches = soup.select('.match-card, .bet-card, .event-card')
            
            for match in matches:
                try:
                    # Extract team names and odds
                    teams = match.select('.team-name, .participant')
                    odds_elements = match.select('.odds, .coefficient')
                    
                    if len(teams) >= 2 and len(odds_elements) >= 2:
                        team1 = teams[0].get_text(strip=True)
                        team2 = teams[1].get_text(strip=True)
                        
                        odds1 = float(re.sub(r'[^\d.]', '', odds_elements[0].get_text()))
                        odds2 = float(re.sub(r'[^\d.]', '', odds_elements[1].get_text()))
                        
                        if self._validate_odds_values([odds1, odds2]):
                            match_id = f"rivalry_{hash(f'{team1}_{team2}')}"
                            
                            odds_data.append(OddsData(
                                match_id=match_id,
                                bookmaker="rivalry",
                                market="match_winner",
                                odds={team1.lower(): odds1, team2.lower(): odds2},
                                timestamp=datetime.utcnow(),
                                source_url=url,
                                confidence=0.8
                            ))
                
                except Exception as e:
                    logger.debug(f"Error processing Rivalry match: {e}")
                    continue
            
            logger.info(f"Scraped {len(odds_data)} matches from Rivalry")
            
        except Exception as e:
            logger.error(f"Error scraping Rivalry: {e}")
        
        return odds_data
    
    def _validate_odds_values(self, odds: List[float]) -> bool:
        """Validate odds values for sanity"""
        try:
            for odd in odds:
                if not (self.min_odds <= odd <= self.max_odds):
                    return False
            
            # Check if odds make sense (implied probability should be reasonable)
            total_implied_prob = sum(1.0 / odd for odd in odds)
            if total_implied_prob < 0.8 or total_implied_prob > 1.5:  # Allow some margin
                return False
            
            return True
            
        except (ValueError, ZeroDivisionError):
            return False
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names for matching across bookmakers"""
        # Remove common prefixes/suffixes
        normalized = re.sub(r'\s+(esports?|gaming|team)\s*$', '', team_name, flags=re.IGNORECASE)
        normalized = re.sub(r'^(team\s+)', '', normalized, flags=re.IGNORECASE)
        
        # Common team name mappings
        mappings = {
            'natus vincere': 'navi',
            'sk gaming': 'sk',
            'fnatic': 'fnatic',
            'astralis': 'astralis',
            'faze clan': 'faze',
            'g2 esports': 'g2',
            'team liquid': 'liquid',
            'cloud9': 'c9'
        }
        
        normalized_lower = normalized.lower().strip()
        return mappings.get(normalized_lower, normalized_lower)
    
    def _match_teams_across_bookmakers(self, odds_list: List[OddsData]) -> Dict[str, List[OddsData]]:
        """Group odds by matching team pairs across bookmakers"""
        matched_groups = {}
        
        for odds in odds_list:
            team_names = list(odds.odds.keys())
            if len(team_names) != 2:
                continue
            
            # Normalize team names
            norm_team1 = self._normalize_team_name(team_names[0])
            norm_team2 = self._normalize_team_name(team_names[1])
            
            # Create a canonical match key (sorted to handle order differences)
            match_key = tuple(sorted([norm_team1, norm_team2]))
            
            if match_key not in matched_groups:
                matched_groups[match_key] = []
            
            matched_groups[match_key].append(odds)
        
        return matched_groups
    
    def _calculate_consensus(self, odds_group: List[OddsData]) -> MarketConsensus:
        """Calculate consensus odds from multiple bookmakers"""
        if not odds_group:
            return None
        
        # Get all unique team names
        all_teams = set()
        for odds in odds_group:
            all_teams.update(odds.odds.keys())
        
        consensus_odds = {}
        best_odds = {}
        spread = {}
        
        for team in all_teams:
            team_odds = []
            bookmaker_odds = []
            
            for odds in odds_group:
                normalized_teams = {self._normalize_team_name(k): v for k, v in odds.odds.items()}
                norm_team = self._normalize_team_name(team)
                
                if norm_team in normalized_teams:
                    odd_value = normalized_teams[norm_team]
                    team_odds.append(odd_value)
                    bookmaker_odds.append((odd_value, odds.bookmaker))
            
            if team_odds:
                # Weighted average (could weight by bookmaker reliability)
                consensus_odds[team] = sum(team_odds) / len(team_odds)
                
                # Best odds
                best_odd, best_bookmaker = max(bookmaker_odds, key=lambda x: x[0])
                best_odds[team] = (best_odd, best_bookmaker)
                
                # Spread
                spread[team] = max(team_odds) - min(team_odds)
        
        # Check for arbitrage
        arbitrage_opportunity = False
        arbitrage_profit = 0.0
        arbitrage_stakes = None
        
        if len(best_odds) >= 2:
            total_inverse_odds = sum(1.0 / odds_info[0] for odds_info in best_odds.values())
            if total_inverse_odds < 1.0:
                arbitrage_opportunity = True
                arbitrage_profit = 1.0 - total_inverse_odds
                
                # Calculate optimal stakes
                total_stake = 1000  # Base amount
                arbitrage_stakes = {}
                for team, (odds_val, _) in best_odds.items():
                    stake = (total_stake / odds_val) / total_inverse_odds
                    arbitrage_stakes[team] = stake / total_stake  # As fraction
        
        return MarketConsensus(
            match_id=odds_group[0].match_id,
            market=odds_group[0].market,
            consensus_odds=consensus_odds,
            best_odds=best_odds,
            spread=spread,
            bookmaker_count=len(odds_group),
            sources=[odds.bookmaker for odds in odds_group],
            timestamp=datetime.utcnow(),
            arbitrage_opportunity=arbitrage_opportunity,
            arbitrage_profit=arbitrage_profit,
            arbitrage_stakes=arbitrage_stakes
        )
    
    async def scrape_all_sources(self, sources: List[BookmakerSource] = None) -> List[MarketConsensus]:
        """Scrape odds from all available sources and create consensus"""
        if sources is None:
            sources = [
                BookmakerSource.ODDSPORTAL,
                BookmakerSource.GGBET,
                BookmakerSource.RIVALRY,
                # BookmakerSource.PINNACLE,  # Requires API access
            ]
        
        # Scrape all sources concurrently
        scraping_tasks = []
        
        for source in sources:
            if source == BookmakerSource.ODDSPORTAL:
                scraping_tasks.append(self.scrape_oddsportal())
            elif source == BookmakerSource.GGBET:
                scraping_tasks.append(self.scrape_ggbet())
            elif source == BookmakerSource.RIVALRY:
                scraping_tasks.append(self.scrape_rivalry())
            elif source == BookmakerSource.PINNACLE:
                scraping_tasks.append(self.scrape_pinnacle_api())
        
        # Execute all scraping tasks
        results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
        
        # Combine all odds data
        all_odds = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Scraping failed for {sources[i].value}: {result}")
            elif isinstance(result, list):
                all_odds.extend(result)
        
        logger.info(f"Total odds scraped: {len(all_odds)} from {len(sources)} sources")
        
        # Group by matching teams
        matched_groups = self._match_teams_across_bookmakers(all_odds)
        
        # Calculate consensus for each group
        consensus_list = []
        for match_key, odds_group in matched_groups.items():
            if len(odds_group) >= 2:  # Need at least 2 bookmakers for consensus
                consensus = self._calculate_consensus(odds_group)
                if consensus:
                    consensus_list.append(consensus)
        
        # Sort by arbitrage opportunities first, then by bookmaker count
        consensus_list.sort(key=lambda x: (x.arbitrage_opportunity, x.bookmaker_count), reverse=True)
        
        logger.info(f"Created consensus for {len(consensus_list)} matches")
        
        # Log arbitrage opportunities
        arb_opportunities = [c for c in consensus_list if c.arbitrage_opportunity]
        if arb_opportunities:
            logger.info(f"Found {len(arb_opportunities)} arbitrage opportunities!")
            for arb in arb_opportunities:
                logger.info(f"Arbitrage: {arb.match_id} - Profit: {arb.arbitrage_profit:.2%}")
        
        return consensus_list
    
    async def get_cached_odds(self, match_id: str) -> Optional[List[OddsData]]:
        """Get cached odds for a match"""
        if match_id in self.odds_cache:
            cached_odds = self.odds_cache[match_id]
            # Check if cache is still valid
            if cached_odds and (datetime.utcnow() - cached_odds[0].timestamp).seconds < self.cache_ttl:
                return cached_odds
        return None
    
    def clear_cache(self):
        """Clear odds cache"""
        self.odds_cache.clear()
        logger.info("Odds cache cleared")
    
    async def get_odds_for_match(self, team1: str, team2: str) -> Optional[MarketConsensus]:
        """Get consensus odds for a specific match"""
        consensus_list = await self.scrape_all_sources()
        
        # Find matching consensus
        norm_team1 = self._normalize_team_name(team1)
        norm_team2 = self._normalize_team_name(team2)
        match_key = tuple(sorted([norm_team1, norm_team2]))
        
        for consensus in consensus_list:
            consensus_teams = set(self._normalize_team_name(team) for team in consensus.consensus_odds.keys())
            if consensus_teams == set(match_key):
                return consensus
        
        return None


# Usage example
async def main():
    """Example usage of Robust Odds Scraper"""
    async with RobustOddsScraper() as scraper:
        # Get consensus odds from all sources
        consensus_list = await scraper.scrape_all_sources()
        
        print(f"\n=== Found {len(consensus_list)} matches with consensus odds ===")
        
        for consensus in consensus_list[:5]:  # Show top 5
            print(f"\nMatch: {consensus.match_id}")
            print(f"Bookmakers: {consensus.bookmaker_count} ({', '.join(consensus.sources)})")
            print(f"Consensus odds: {consensus.consensus_odds}")
            print(f"Best odds: {consensus.best_odds}")
            
            if consensus.arbitrage_opportunity:
                print(f"🚨 ARBITRAGE: {consensus.arbitrage_profit:.2%} profit!")
                print(f"Stakes: {consensus.arbitrage_stakes}")


if __name__ == "__main__":
    asyncio.run(main())
