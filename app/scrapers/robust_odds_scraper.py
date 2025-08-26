#!/usr/bin/env python3
"""
Robust Multi-Source Odds Scraper
Fetches odds from multiple bookmakers with fallback, validation, and retry logic
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import re
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import backoff

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
    """Production-ready multi-source odds scraper"""
    
    def __init__(self, max_concurrent: int = 3, timeout: int = 15):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.ua = UserAgent()
        
        # Session management
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        
        # Rate limiting per source
        self.rate_limits = {
            BookmakerSource.ODDSPORTAL: {'requests': 0, 'window_start': 0, 'max_per_minute': 20},
            BookmakerSource.BET365: {'requests': 0, 'window_start': 0, 'max_per_minute': 15},
            BookmakerSource.PINNACLE: {'requests': 0, 'window_start': 0, 'max_per_minute': 30},
            BookmakerSource.BETWAY: {'requests': 0, 'window_start': 0, 'max_per_minute': 25},
            BookmakerSource.UNIBET: {'requests': 0, 'window_start': 0, 'max_per_minute': 20},
            BookmakerSource.GGBET: {'requests': 0, 'window_start': 0, 'max_per_minute': 15},
            BookmakerSource.RIVALRY: {'requests': 0, 'window_start': 0, 'max_per_minute': 20},
        }
        
        # Caching
        self.odds_cache: Dict[str, List[OddsData]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Validation thresholds
        self.min_odds = 1.01
        self.max_odds = 50.0
        self.max_spread_threshold = 0.5  # 50% spread is suspicious
        
        # Bookmaker configurations
        self.bookmaker_configs = {
            BookmakerSource.ODDSPORTAL: {
                'base_url': 'https://www.oddsportal.com',
                'selectors': {
                    'match_rows': '.table-main tbody tr',
                    'team_names': '.name',
                    'odds': '.odds-nowrp',
                    'match_time': '.table-time'
                },
                'headers': {
                    'User-Agent': self.ua.random,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            },
            BookmakerSource.BET365: {
                'base_url': 'https://www.bet365.com',
                'selectors': {
                    'match_rows': '.gl-Market_General',
                    'team_names': '.gl-Participant_Name',
                    'odds': '.gl-Participant_Odds'
                }
            },
            BookmakerSource.PINNACLE: {
                'base_url': 'https://www.pinnacle.com',
                'api_endpoint': '/api/v1/odds',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
            },
            BookmakerSource.GGBET: {
                'base_url': 'https://gg.bet',
                'selectors': {
                    'match_rows': '.match-item',
                    'team_names': '.team-name',
                    'odds': '.odds-value'
                }
            }
        }
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self):
        """Initialize HTTP sessions for each bookmaker"""
        for source in BookmakerSource:
            config = self.bookmaker_configs.get(source, {})
            headers = config.get('headers', {
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            })
            
            connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)
            timeout = aiohttp.ClientTimeout(total=self.timeout, connect=5)
            
            self.sessions[source.value] = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
        
        logger.info(f"Initialized {len(self.sessions)} bookmaker sessions")
    
    async def close(self):
        """Close all HTTP sessions"""
        for session in self.sessions.values():
            await session.close()
        self.sessions.clear()
    
    async def _rate_limit_check(self, source: BookmakerSource):
        """Check and enforce rate limits"""
        current_time = asyncio.get_event_loop().time()
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
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _fetch_with_retry(
        self, 
        source: BookmakerSource, 
        url: str, 
        **kwargs
    ) -> Optional[aiohttp.ClientResponse]:
        """Fetch URL with exponential backoff retry"""
        await self._rate_limit_check(source)
        
        session = self.sessions.get(source.value)
        if not session:
            logger.error(f"No session available for {source.value}")
            return None
        
        try:
            async with session.get(url, **kwargs) as response:
                if response.status == 200:
                    return response
                elif response.status == 429:
                    # Rate limited, wait longer
                    await asyncio.sleep(5)
                    raise aiohttp.ClientError("Rate limited")
                elif response.status >= 500:
                    # Server error, retry
                    raise aiohttp.ClientError(f"Server error: {response.status}")
                else:
                    logger.warning(f"HTTP {response.status} from {source.value}: {url}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching from {source.value}: {url}")
            raise
        except Exception as e:
            logger.error(f"Error fetching from {source.value}: {e}")
            raise
    
    async def scrape_oddsportal(self, sport: str = "esports") -> List[OddsData]:
        """Scrape odds from OddsPortal"""
        odds_data = []
        
        try:
            url = f"https://www.oddsportal.com/{sport}/counter-strike/"
            response = await self._fetch_with_retry(BookmakerSource.ODDSPORTAL, url)
            
            if not response:
                return odds_data
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
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
            response = await self._fetch_with_retry(BookmakerSource.GGBET, url)
            
            if not response:
                return odds_data
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
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
            response = await self._fetch_with_retry(BookmakerSource.RIVALRY, url)
            
            if not response:
                return odds_data
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
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
