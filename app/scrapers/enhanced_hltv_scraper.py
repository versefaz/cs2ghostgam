#!/usr/bin/env python3
"""
Enhanced HLTV Scraper - Complete Team Statistics
Fetches comprehensive team data: rankings, recent form, head-to-head, map statistics
"""

import asyncio
import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

import aiohttp
import asyncio
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Comprehensive team statistics"""
    team_id: str
    team_name: str
    world_ranking: int
    ranking_points: int
    recent_form: List[str]  # W/L for last 10 matches
    win_rate_30d: float
    maps_played_30d: int
    
    # Map statistics
    map_stats: Dict[str, Dict[str, Any]]  # map_name -> {wins, losses, win_rate}
    
    # Recent performance
    avg_rating_30d: float
    kd_ratio_30d: float
    
    # Head-to-head vs specific opponent
    h2h_wins: int = 0
    h2h_losses: int = 0
    h2h_last_match: Optional[str] = None
    h2h_win_rate: float = 0.0
    
    # Additional metadata
    country: str = ""
    logo_url: str = ""
    last_updated: datetime = None


@dataclass
class MatchContext:
    """Enhanced match context with team statistics"""
    match_id: str
    team1_name: str
    team2_name: str
    team1_stats: TeamStats
    team2_stats: TeamStats
    event_name: str
    match_time: datetime
    bo_format: str  # BO1, BO3, BO5
    
    # Derived insights
    ranking_advantage: int  # team1_ranking - team2_ranking
    form_advantage: float  # team1_form - team2_form
    h2h_advantage: float  # team1_h2h_wr - team2_h2h_wr


class EnhancedHLTVScraper:
    """Production-ready HLTV scraper with comprehensive team data"""
    
    def __init__(self, max_concurrent: int = 5, delay_range: Tuple[float, float] = (1.0, 3.0)):
        self.base_url = "https://www.hltv.org"
        self.max_concurrent = max_concurrent
        self.delay_range = delay_range
        self.ua = UserAgent()
        
        # Caching
        self.team_cache: Dict[str, TeamStats] = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_window = 30
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self):
        """Initialize HTTP session"""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=3)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
        
        logger.info("Enhanced HLTV Scraper initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = asyncio.get_event_loop().time()
        
        # Reset counter if window passed
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Check rate limit
        if self.request_count >= self.max_requests_per_window:
            sleep_time = self.rate_limit_window - (current_time - self.last_request_time)
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                self.request_count = 0
        
        # Random delay between requests
        delay = self.delay_range[0] + (self.delay_range[1] - self.delay_range[0]) * asyncio.get_event_loop().time() % 1
        await asyncio.sleep(delay)
        
        self.request_count += 1
    
    async def _fetch_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML page with retries"""
        if not self.session:
            await self.initialize()
        
        for attempt in range(retries):
            try:
                await self._rate_limit()
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        return BeautifulSoup(html, 'html.parser')
                    elif response.status == 429:
                        # Rate limited, wait longer
                        wait_time = 2 ** attempt * 5
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return None
    
    async def get_team_ranking(self) -> Dict[str, Dict[str, Any]]:
        """Fetch current world rankings"""
        try:
            url = f"{self.base_url}/ranking/teams"
            soup = await self._fetch_page(url)
            
            if not soup:
                return {}
            
            rankings = {}
            
            # Parse ranking table
            for i, row in enumerate(soup.select('.ranked-team'), 1):
                try:
                    team_link = row.select_one('.teamLine a')
                    if not team_link:
                        continue
                    
                    team_name = team_link.text.strip()
                    team_id = team_link.get('href', '').split('/')[-2] if team_link.get('href') else str(i)
                    
                    points_elem = row.select_one('.points')
                    points = int(points_elem.text.strip()) if points_elem else 0
                    
                    rankings[team_name.lower()] = {
                        'team_id': team_id,
                        'team_name': team_name,
                        'ranking': i,
                        'points': points
                    }
                    
                except Exception as e:
                    logger.error(f"Error parsing ranking row: {e}")
                    continue
            
            logger.info(f"Fetched rankings for {len(rankings)} teams")
            return rankings
            
        except Exception as e:
            logger.error(f"Error fetching team rankings: {e}")
            return {}
    
    async def get_team_stats(self, team_name: str, team_id: str = None) -> Optional[TeamStats]:
        """Fetch comprehensive team statistics"""
        try:
            # Check cache first
            cache_key = team_name.lower()
            if cache_key in self.team_cache:
                cached_stats = self.team_cache[cache_key]
                if cached_stats.last_updated and (datetime.utcnow() - cached_stats.last_updated).seconds < self.cache_ttl:
                    return cached_stats
            
            # Find team ID if not provided
            if not team_id:
                team_id = await self._find_team_id(team_name)
                if not team_id:
                    logger.warning(f"Could not find team ID for {team_name}")
                    return None
            
            # Fetch team page
            team_url = f"{self.base_url}/team/{team_id}/{team_name.replace(' ', '-').lower()}"
            soup = await self._fetch_page(team_url)
            
            if not soup:
                return None
            
            # Parse team statistics
            stats = await self._parse_team_page(soup, team_name, team_id)
            
            # Fetch recent matches for form
            recent_matches = await self._get_team_recent_matches(team_id, limit=10)
            if recent_matches:
                stats.recent_form = [match['result'] for match in recent_matches]
                stats.win_rate_30d = sum(1 for r in stats.recent_form if r == 'W') / len(stats.recent_form)
                stats.maps_played_30d = len(recent_matches)
            
            # Fetch map statistics
            map_stats = await self._get_team_map_stats(team_id)
            if map_stats:
                stats.map_stats = map_stats
            
            # Cache results
            stats.last_updated = datetime.utcnow()
            self.team_cache[cache_key] = stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching team stats for {team_name}: {e}")
            return None
    
    async def _find_team_id(self, team_name: str) -> Optional[str]:
        """Find team ID by searching"""
        try:
            search_url = f"{self.base_url}/search?term={team_name.replace(' ', '+')}"
            soup = await self._fetch_page(search_url)
            
            if not soup:
                return None
            
            # Look for team results
            for result in soup.select('.search-result'):
                if 'team' in result.get('class', []):
                    link = result.select_one('a')
                    if link and team_name.lower() in link.text.lower():
                        href = link.get('href', '')
                        if '/team/' in href:
                            return href.split('/')[2]
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding team ID for {team_name}: {e}")
            return None
    
    async def _parse_team_page(self, soup: BeautifulSoup, team_name: str, team_id: str) -> TeamStats:
        """Parse team page for basic information"""
        try:
            # Basic info
            ranking_elem = soup.select_one('.profile-team-stat .right')
            world_ranking = int(ranking_elem.text.strip().replace('#', '')) if ranking_elem else 999
            
            # Country
            country_elem = soup.select_one('.team-country')
            country = country_elem.get('title', '') if country_elem else ''
            
            # Logo
            logo_elem = soup.select_one('.teamlogo')
            logo_url = logo_elem.get('src', '') if logo_elem else ''
            
            # Points (from ranking page data if available)
            ranking_points = 0
            
            return TeamStats(
                team_id=team_id,
                team_name=team_name,
                world_ranking=world_ranking,
                ranking_points=ranking_points,
                recent_form=[],
                win_rate_30d=0.0,
                maps_played_30d=0,
                map_stats={},
                avg_rating_30d=0.0,
                kd_ratio_30d=0.0,
                country=country,
                logo_url=logo_url
            )
            
        except Exception as e:
            logger.error(f"Error parsing team page: {e}")
            return TeamStats(
                team_id=team_id,
                team_name=team_name,
                world_ranking=999,
                ranking_points=0,
                recent_form=[],
                win_rate_30d=0.0,
                maps_played_30d=0,
                map_stats={},
                avg_rating_30d=0.0,
                kd_ratio_30d=0.0
            )
    
    async def _get_team_recent_matches(self, team_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch team's recent match results"""
        try:
            matches_url = f"{self.base_url}/team/{team_id}/matches"
            soup = await self._fetch_page(matches_url)
            
            if not soup:
                return []
            
            matches = []
            
            for match_row in soup.select('.results-holder .result-con')[:limit]:
                try:
                    # Match result
                    result_elem = match_row.select_one('.result-score')
                    if not result_elem:
                        continue
                    
                    # Determine win/loss
                    score_text = result_elem.text.strip()
                    if '-' in score_text:
                        scores = score_text.split('-')
                        if len(scores) == 2:
                            score1, score2 = map(int, scores)
                            result = 'W' if score1 > score2 else 'L'
                        else:
                            result = 'L'  # Default to loss if unclear
                    else:
                        result = 'L'
                    
                    # Match date
                    date_elem = match_row.select_one('.match-day')
                    match_date = date_elem.text.strip() if date_elem else ''
                    
                    # Opponent
                    opponent_elem = match_row.select_one('.team')
                    opponent = opponent_elem.text.strip() if opponent_elem else 'Unknown'
                    
                    matches.append({
                        'result': result,
                        'score': score_text,
                        'opponent': opponent,
                        'date': match_date
                    })
                    
                except Exception as e:
                    logger.error(f"Error parsing match row: {e}")
                    continue
            
            return matches
            
        except Exception as e:
            logger.error(f"Error fetching recent matches for team {team_id}: {e}")
            return []
    
    async def _get_team_map_stats(self, team_id: str) -> Dict[str, Dict[str, Any]]:
        """Fetch team's map statistics"""
        try:
            maps_url = f"{self.base_url}/team/{team_id}/maps"
            soup = await self._fetch_page(maps_url)
            
            if not soup:
                return {}
            
            map_stats = {}
            
            # Parse map statistics table
            for row in soup.select('.stats-table tbody tr'):
                try:
                    map_elem = row.select_one('.map-pool-map-name')
                    if not map_elem:
                        continue
                    
                    map_name = map_elem.text.strip()
                    
                    # Get statistics columns
                    cols = row.select('td')
                    if len(cols) >= 4:
                        maps_played = int(cols[1].text.strip()) if cols[1].text.strip().isdigit() else 0
                        wins = int(cols[2].text.strip()) if cols[2].text.strip().isdigit() else 0
                        win_rate = float(cols[3].text.strip().replace('%', '')) / 100 if '%' in cols[3].text else 0.0
                        
                        losses = maps_played - wins
                        
                        map_stats[map_name] = {
                            'maps_played': maps_played,
                            'wins': wins,
                            'losses': losses,
                            'win_rate': win_rate
                        }
                
                except Exception as e:
                    logger.error(f"Error parsing map stats row: {e}")
                    continue
            
            return map_stats
            
        except Exception as e:
            logger.error(f"Error fetching map stats for team {team_id}: {e}")
            return {}
    
    async def get_head_to_head(self, team1_name: str, team2_name: str) -> Dict[str, Any]:
        """Fetch head-to-head statistics between two teams"""
        try:
            # This would require searching for matches between specific teams
            # For now, return empty data - to be implemented with match history search
            return {
                'team1_wins': 0,
                'team2_wins': 0,
                'total_matches': 0,
                'last_match_date': None,
                'last_match_result': None,
                'recent_h2h': []  # Last 5 matches
            }
            
        except Exception as e:
            logger.error(f"Error fetching H2H for {team1_name} vs {team2_name}: {e}")
            return {}
    
    async def get_enhanced_match_data(self, match_id: str) -> Optional[MatchContext]:
        """Fetch comprehensive match data with team statistics"""
        try:
            # First get basic match info
            match_url = f"{self.base_url}/matches/{match_id}"
            soup = await self._fetch_page(match_url)
            
            if not soup:
                return None
            
            # Parse match details
            team_elements = soup.select('.teamName')
            if len(team_elements) < 2:
                logger.warning(f"Could not find both teams for match {match_id}")
                return None
            
            team1_name = team_elements[0].text.strip()
            team2_name = team_elements[1].text.strip()
            
            # Event name
            event_elem = soup.select_one('.event-name')
            event_name = event_elem.text.strip() if event_elem else 'Unknown Event'
            
            # Match time
            time_elem = soup.select_one('.time')
            match_time = datetime.utcnow()  # Default to now if not found
            
            # BO format
            bo_elem = soup.select_one('.preformatted-text')
            bo_format = bo_elem.text.strip() if bo_elem else 'BO1'
            
            # Fetch team statistics
            team1_stats, team2_stats = await asyncio.gather(
                self.get_team_stats(team1_name),
                self.get_team_stats(team2_name)
            )
            
            if not team1_stats or not team2_stats:
                logger.warning(f"Could not fetch team stats for match {match_id}")
                return None
            
            # Fetch head-to-head
            h2h_data = await self.get_head_to_head(team1_name, team2_name)
            
            # Update team stats with H2H data
            if h2h_data:
                team1_stats.h2h_wins = h2h_data.get('team1_wins', 0)
                team1_stats.h2h_losses = h2h_data.get('team2_wins', 0)
                team2_stats.h2h_wins = h2h_data.get('team2_wins', 0)
                team2_stats.h2h_losses = h2h_data.get('team1_wins', 0)
                
                total_h2h = team1_stats.h2h_wins + team1_stats.h2h_losses
                if total_h2h > 0:
                    team1_stats.h2h_win_rate = team1_stats.h2h_wins / total_h2h
                    team2_stats.h2h_win_rate = team2_stats.h2h_wins / total_h2h
            
            # Calculate derived insights
            ranking_advantage = team2_stats.world_ranking - team1_stats.world_ranking  # Lower ranking number is better
            form_advantage = team1_stats.win_rate_30d - team2_stats.win_rate_30d
            h2h_advantage = team1_stats.h2h_win_rate - team2_stats.h2h_win_rate
            
            return MatchContext(
                match_id=match_id,
                team1_name=team1_name,
                team2_name=team2_name,
                team1_stats=team1_stats,
                team2_stats=team2_stats,
                event_name=event_name,
                match_time=match_time,
                bo_format=bo_format,
                ranking_advantage=ranking_advantage,
                form_advantage=form_advantage,
                h2h_advantage=h2h_advantage
            )
            
        except Exception as e:
            logger.error(f"Error fetching enhanced match data for {match_id}: {e}")
            return None
    
    async def get_upcoming_matches_with_stats(self, limit: int = 50) -> List[MatchContext]:
        """Fetch upcoming matches with comprehensive team statistics"""
        try:
            matches_url = f"{self.base_url}/matches"
            soup = await self._fetch_page(matches_url)
            
            if not soup:
                return []
            
            match_contexts = []
            
            # Parse upcoming matches
            for match_elem in soup.select('.upcomingMatch')[:limit]:
                try:
                    # Extract match ID
                    match_id = match_elem.get('data-zonedgrouping-entry-unix') or \
                              match_elem.get('data-match-id') or \
                              str(hash(match_elem.text))
                    
                    # Get enhanced match data
                    match_context = await self.get_enhanced_match_data(match_id)
                    if match_context:
                        match_contexts.append(match_context)
                    
                    # Respect rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing upcoming match: {e}")
                    continue
            
            logger.info(f"Fetched {len(match_contexts)} upcoming matches with stats")
            return match_contexts
            
        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {e}")
            return []
    
    def get_cached_team_stats(self, team_name: str) -> Optional[TeamStats]:
        """Get team stats from cache"""
        return self.team_cache.get(team_name.lower())
    
    def clear_cache(self):
        """Clear team statistics cache"""
        self.team_cache.clear()
        logger.info("Team stats cache cleared")
    
    async def warm_cache(self, team_names: List[str]):
        """Pre-load team statistics into cache"""
        logger.info(f"Warming cache for {len(team_names)} teams")
        
        # Fetch team rankings first
        rankings = await self.get_team_ranking()
        
        # Fetch team stats concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_team_stats(team_name: str):
            async with semaphore:
                team_info = rankings.get(team_name.lower())
                team_id = team_info['team_id'] if team_info else None
                return await self.get_team_stats(team_name, team_id)
        
        tasks = [fetch_team_stats(team) for team in team_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if isinstance(r, TeamStats))
        logger.info(f"Cache warmed: {successful}/{len(team_names)} teams loaded")


# Usage example and integration
async def main():
    """Example usage of Enhanced HLTV Scraper"""
    async with EnhancedHLTVScraper() as scraper:
        # Get upcoming matches with full stats
        matches = await scraper.get_upcoming_matches_with_stats(limit=10)
        
        for match in matches:
            print(f"\n=== {match.team1_name} vs {match.team2_name} ===")
            print(f"Event: {match.event_name}")
            print(f"Format: {match.bo_format}")
            print(f"Ranking: #{match.team1_stats.world_ranking} vs #{match.team2_stats.world_ranking}")
            print(f"Recent Form: {match.team1_stats.win_rate_30d:.1%} vs {match.team2_stats.win_rate_30d:.1%}")
            print(f"Advantages: Ranking={match.ranking_advantage}, Form={match.form_advantage:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
