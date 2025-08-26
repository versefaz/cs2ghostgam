import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import re
from concurrent.futures import ThreadPoolExecutor

# Optional dependencies with safe imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import undetected_chromedriver as uc
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)


class HLTVStatsScraper:
    """
    ระบบดึงข้อมูลสถิติจาก HLTV แบบ High-Performance
    ใช้ multi-method scraping และ caching strategy
    """
    
    def __init__(self):
        self.base_url = "https://www.hltv.org"
        self.session = None
        self.driver = None
        self.cache = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize cache if Redis available
        if REDIS_AVAILABLE:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                self.cache.ping()  # Test connection
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.cache = None
        
        # Initialize cloudscraper if available
        if CLOUDSCRAPER_AVAILABLE:
            self.scraper = cloudscraper.create_scraper()
        else:
            self.scraper = None
        
        # Rate limiting
        self.last_request = {}
        self.min_delay = 1.0  # seconds between requests
        
        # Headers สำหรับ bypass detection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
    async def initialize(self):
        """Initialize async session และ selenium driver"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        
        # Setup undetected Chrome driver if available
        if SELENIUM_AVAILABLE:
            try:
                options = uc.ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                
                self.driver = uc.Chrome(options=options)
                logger.info("Selenium driver initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Selenium: {e}")
                
    async def get_team_stats(self, team_name: str, force_refresh: bool = False) -> Dict:
        """
        ดึงข้อมูลสถิติทีมแบบครบถ้วน
        
        Returns:
            - World ranking
            - Recent form (last 3 months)
            - Map statistics
            - Player ratings
            - Trophy cabinet
        """
        cache_key = f"hltv:team:{team_name}"
        
        # Check cache first
        if not force_refresh and self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    logger.info(f"Using cached data for {team_name}")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        logger.info(f"Fetching fresh data for {team_name}")
        
        # Search for team ID
        team_id = await self._get_team_id(team_name)
        if not team_id:
            logger.error(f"Team {team_name} not found")
            return {}
        
        # Parallel fetch all stats
        tasks = [
            self._fetch_team_overview(team_id),
            self._fetch_team_matches(team_id, days=90),
            self._fetch_map_stats(team_id),
            self._fetch_player_stats(team_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all stats
        team_stats = {
            'team_name': team_name,
            'team_id': team_id,
            'last_updated': datetime.now().isoformat(),
            'overview': results[0] if not isinstance(results[0], Exception) else {},
            'recent_matches': results[1] if not isinstance(results[1], Exception) else [],
            'map_stats': results[2] if not isinstance(results[2], Exception) else {},
            'player_stats': results[3] if not isinstance(results[3], Exception) else []
        }
        
        # Process and calculate derived metrics
        team_stats['processed'] = self._process_team_metrics(team_stats)
        
        # Cache for 1 hour
        if self.cache:
            try:
                self.cache.setex(cache_key, 3600, json.dumps(team_stats))
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        return team_stats
    
    async def _get_team_id(self, team_name: str) -> Optional[int]:
        """ค้นหา team ID จากชื่อทีม"""
        # Clean team name
        search_name = team_name.lower().replace(' ', '-')
        
        # Try direct URL first
        url = f"{self.base_url}/team/{search_name}"
        
        try:
            response = await self._fetch_with_retry(url)
            if response and hasattr(response, 'url') and 'team/' in str(response.url):
                # Extract team ID from URL
                match = re.search(r'/team/(\d+)/', str(response.url))
                if match:
                    return int(match.group(1))
        except:
            pass
        
        # Search via HLTV search
        search_url = f"{self.base_url}/search?query={team_name}"
        try:
            response = await self._fetch_with_retry(search_url)
            if response:
                soup = BeautifulSoup(response, 'html.parser')
                
                # Find team in search results
                team_links = soup.find_all('a', href=re.compile(r'/team/\d+/'))
                for link in team_links:
                    if team_name.lower() in link.text.lower():
                        match = re.search(r'/team/(\d+)/', link['href'])
                        if match:
                            return int(match.group(1))
        except Exception as e:
            logger.error(f"Error searching team ID: {e}")
        
        return None
    
    async def _fetch_team_overview(self, team_id: int) -> Dict:
        """ดึงข้อมูล overview ของทีม"""
        url = f"{self.base_url}/team/{team_id}/"
        
        try:
            response = await self._fetch_with_retry(url)
            if not response:
                return {}
                
            soup = BeautifulSoup(response, 'html.parser')
            
            overview = {}
            
            # World ranking
            ranking = soup.find('div', class_='profile-team-stat')
            if ranking:
                rank_text = ranking.find('span', class_='value')
                if rank_text:
                    rank_match = re.search(r'\d+', rank_text.text)
                    if rank_match:
                        overview['world_ranking'] = int(rank_match.group())
            
            # Team info
            team_info = soup.find('div', class_='profile-team-info')
            if team_info:
                flag = team_info.find('img', class_='flag')
                overview['country'] = flag.get('title', 'Unknown') if flag else 'Unknown'
                
            # Stats summary
            stats_container = soup.find('div', class_='profile-team-stats-container')
            if stats_container:
                stat_values = stats_container.find_all('div', class_='stat-value')
                if len(stat_values) >= 3:
                    try:
                        age_match = re.search(r'[\d.]+', stat_values[0].text)
                        if age_match:
                            overview['avg_player_age'] = float(age_match.group())
                        
                        weeks_match = re.search(r'\d+', stat_values[1].text)
                        if weeks_match:
                            overview['weeks_in_top30'] = int(weeks_match.group())
                        
                        trophies_match = re.search(r'\d+', stat_values[2].text)
                        if trophies_match:
                            overview['trophies'] = int(trophies_match.group())
                    except (ValueError, AttributeError):
                        pass
            
            # Current lineup
            lineup = []
            players_container = soup.find('div', class_='bodyshot-team')
            if players_container:
                players = players_container.find_all('a', class_='player-link')
                for player in players:
                    player_name = player.find('span', class_='text-ellipsis')
                    if player_name:
                        lineup.append(player_name.text.strip())
            overview['current_lineup'] = lineup
            
            return overview
            
        except Exception as e:
            logger.error(f"Error fetching team overview: {e}")
            return {}
    
    async def _fetch_team_matches(self, team_id: int, days: int = 90) -> List[Dict]:
        """ดึงประวัติการแข่งขันล่าสุด"""
        url = f"{self.base_url}/results?team={team_id}"
        
        try:
            response = await self._fetch_with_retry(url)
            if not response:
                return []
                
            soup = BeautifulSoup(response, 'html.parser')
            
            matches = []
            match_elements = soup.find_all('div', class_='result-con', limit=50)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for match_elem in match_elements:
                try:
                    match_data = self._parse_match_element(match_elem)
                    if match_data and match_data.get('date'):
                        match_date = datetime.fromisoformat(match_data['date'])
                        if match_date >= cutoff_date:
                            matches.append(match_data)
                except Exception as e:
                    logger.debug(f"Error parsing match element: {e}")
                    continue
            
            return matches
            
        except Exception as e:
            logger.error(f"Error fetching team matches: {e}")
            return []
    
    def _parse_match_element(self, match_elem) -> Optional[Dict]:
        """Parse individual match element"""
        try:
            match_data = {}
            
            # Date
            date_elem = match_elem.find('div', class_='date')
            if date_elem:
                match_data['date'] = self._parse_date(date_elem.text.strip())
            
            # Teams and score
            team_elements = match_elem.find_all('div', class_='team')
            if len(team_elements) >= 2:
                match_data['team1'] = team_elements[0].text.strip()
                match_data['team2'] = team_elements[1].text.strip()
            
            # Score
            score_elem = match_elem.find('span', class_='score')
            if score_elem:
                match_data['score'] = score_elem.text.strip()
                # Parse individual scores
                scores = re.findall(r'\d+', score_elem.text)
                if len(scores) >= 2:
                    match_data['team1_score'] = int(scores[0])
                    match_data['team2_score'] = int(scores[1])
            
            # Map
            map_elem = match_elem.find('div', class_='map')
            if map_elem:
                match_data['map'] = map_elem.text.strip()
            
            # Event
            event_elem = match_elem.find('div', class_='event')
            if event_elem:
                match_data['event'] = event_elem.text.strip()
            
            # Determine result
            if 'team1_score' in match_data and 'team2_score' in match_data:
                match_data['result'] = 'win' if match_data['team1_score'] > match_data['team2_score'] else 'loss'
            
            return match_data if match_data else None
            
        except Exception as e:
            logger.debug(f"Error parsing match: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date string to ISO format"""
        try:
            # Handle various date formats from HLTV
            if 'ago' in date_str.lower():
                # Handle relative dates like "2 days ago"
                if 'day' in date_str:
                    days = int(re.search(r'\d+', date_str).group())
                    date = datetime.now() - timedelta(days=days)
                elif 'hour' in date_str:
                    hours = int(re.search(r'\d+', date_str).group())
                    date = datetime.now() - timedelta(hours=hours)
                else:
                    date = datetime.now()
                return date.isoformat()
            else:
                # Try to parse absolute dates
                # This is a simplified parser - HLTV uses various formats
                return datetime.now().isoformat()
        except:
            return datetime.now().isoformat()
    
    async def _fetch_map_stats(self, team_id: int) -> Dict:
        """ดึงสถิติของทีมในแต่ละแมพ"""
        url = f"{self.base_url}/stats/teams/maps/{team_id}"
        
        try:
            response = await self._fetch_with_retry(url)
            if not response:
                return {}
                
            soup = BeautifulSoup(response, 'html.parser')
            
            map_stats = {}
            
            # Find map statistics table
            stats_table = soup.find('table', class_='stats-table')
            if stats_table:
                rows = stats_table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        map_name = cells[0].text.strip()
                        matches = int(cells[1].text.strip()) if cells[1].text.strip().isdigit() else 0
                        wins = int(cells[2].text.strip()) if cells[2].text.strip().isdigit() else 0
                        win_rate = float(cells[3].text.strip().replace('%', '')) / 100 if '%' in cells[3].text else 0.0
                        
                        map_stats[map_name] = {
                            'matches': matches,
                            'wins': wins,
                            'win_rate': win_rate
                        }
            
            return map_stats
            
        except Exception as e:
            logger.error(f"Error fetching map stats: {e}")
            return {}
    
    async def _fetch_player_stats(self, team_id: int) -> List[Dict]:
        """ดึงสถิติผู้เล่นในทีม"""
        url = f"{self.base_url}/stats/teams/{team_id}"
        
        try:
            response = await self._fetch_with_retry(url)
            if not response:
                return []
                
            soup = BeautifulSoup(response, 'html.parser')
            
            players = []
            
            # Find player statistics table
            player_table = soup.find('table', class_='player-table')
            if player_table:
                rows = player_table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 6:
                        player_data = {
                            'name': cells[0].text.strip(),
                            'maps': int(cells[1].text.strip()) if cells[1].text.strip().isdigit() else 0,
                            'rating': float(cells[2].text.strip()) if cells[2].text.strip().replace('.', '').isdigit() else 0.0,
                            'kd_ratio': float(cells[3].text.strip()) if cells[3].text.strip().replace('.', '').isdigit() else 0.0,
                            'adr': float(cells[4].text.strip()) if cells[4].text.strip().replace('.', '').isdigit() else 0.0,
                            'kast': float(cells[5].text.strip().replace('%', '')) / 100 if '%' in cells[5].text else 0.0
                        }
                        players.append(player_data)
            
            return players
            
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            return []
    
    def _process_team_metrics(self, team_stats: Dict) -> Dict:
        """Process และคำนวณ derived metrics"""
        processed = {}
        
        # Calculate recent form
        matches = team_stats.get('recent_matches', [])
        if matches:
            wins = sum(1 for m in matches if m.get('result') == 'win')
            total = len(matches)
            processed['recent_win_rate'] = wins / total if total > 0 else 0.0
            processed['recent_matches_count'] = total
            
            # Calculate momentum (weighted recent performance)
            if total > 0:
                weights = [0.9 ** i for i in range(total)]  # More recent = higher weight
                win_values = [1 if m.get('result') == 'win' else 0 for m in matches]
                momentum = sum(w * v for w, v in zip(weights, win_values)) / sum(weights)
                processed['momentum'] = momentum
        
        # Map pool analysis
        map_stats = team_stats.get('map_stats', {})
        if map_stats:
            # Find strongest and weakest maps
            sorted_maps = sorted(map_stats.items(), key=lambda x: x[1].get('win_rate', 0), reverse=True)
            processed['strongest_map'] = sorted_maps[0][0] if sorted_maps else None
            processed['weakest_map'] = sorted_maps[-1][0] if sorted_maps else None
            
            # Calculate average map performance
            total_matches = sum(m.get('matches', 0) for m in map_stats.values())
            weighted_winrate = sum(m.get('win_rate', 0) * m.get('matches', 0) for m in map_stats.values())
            processed['avg_map_winrate'] = weighted_winrate / total_matches if total_matches > 0 else 0.0
        
        # Player analysis
        players = team_stats.get('player_stats', [])
        if players:
            ratings = [p.get('rating', 0) for p in players]
            processed['avg_team_rating'] = sum(ratings) / len(ratings) if ratings else 0.0
            processed['star_player'] = max(players, key=lambda p: p.get('rating', 0))['name'] if players else None
        
        return processed
    
    async def _fetch_with_retry(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Fetch URL with multiple methods and retry logic"""
        
        # Rate limiting
        await self._rate_limit(url)
        
        methods = []
        
        # Add available methods
        if self.session:
            methods.append(self._fetch_with_aiohttp)
        if self.scraper:
            methods.append(self._fetch_with_cloudscraper)
        if self.driver:
            methods.append(self._fetch_with_selenium)
        
        # Fallback to basic requests
        methods.append(self._fetch_with_basic)
        
        for method in methods:
            for attempt in range(max_retries):
                try:
                    result = await method(url)
                    if result:
                        return result
                except Exception as e:
                    logger.debug(f"Method {method.__name__} attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
        
        logger.error(f"All methods failed for URL: {url}")
        return None
    
    async def _fetch_with_aiohttp(self, url: str) -> Optional[str]:
        """Fetch using aiohttp"""
        if not self.session:
            return None
            
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.text()
        return None
    
    async def _fetch_with_cloudscraper(self, url: str) -> Optional[str]:
        """Fetch using cloudscraper"""
        if not self.scraper:
            return None
            
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(self.executor, self.scraper.get, url)
        
        if response.status_code == 200:
            return response.text
        return None
    
    async def _fetch_with_selenium(self, url: str) -> Optional[str]:
        """Fetch using Selenium"""
        if not self.driver:
            return None
            
        loop = asyncio.get_event_loop()
        
        def selenium_get():
            self.driver.get(url)
            return self.driver.page_source
        
        return await loop.run_in_executor(self.executor, selenium_get)
    
    async def _fetch_with_basic(self, url: str) -> Optional[str]:
        """Basic fallback fetch"""
        try:
            import urllib.request
            
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            logger.debug(f"Basic fetch failed: {e}")
            return None
    
    async def _rate_limit(self, url: str):
        """Implement rate limiting"""
        domain = url.split('/')[2]
        now = asyncio.get_event_loop().time()
        
        if domain in self.last_request:
            elapsed = now - self.last_request[domain]
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
        
        self.last_request[domain] = now
    
    async def get_live_matches(self) -> List[Dict]:
        """ดึงข้อมูลแมตช์ที่กำลังแข่งขัน"""
        url = f"{self.base_url}/matches"
        
        try:
            response = await self._fetch_with_retry(url)
            if not response:
                return []
                
            soup = BeautifulSoup(response, 'html.parser')
            
            live_matches = []
            match_elements = soup.find_all('div', class_='upcomingMatch')
            
            for match_elem in match_elements:
                try:
                    match_data = {}
                    
                    # Teams
                    teams = match_elem.find_all('div', class_='matchTeam')
                    if len(teams) >= 2:
                        match_data['team1'] = teams[0].text.strip()
                        match_data['team2'] = teams[1].text.strip()
                    
                    # Time
                    time_elem = match_elem.find('div', class_='matchTime')
                    if time_elem:
                        match_data['time'] = time_elem.text.strip()
                    
                    # Event
                    event_elem = match_elem.find('div', class_='matchEvent')
                    if event_elem:
                        match_data['event'] = event_elem.text.strip()
                    
                    # Format
                    format_elem = match_elem.find('div', class_='matchMeta')
                    if format_elem:
                        match_data['format'] = format_elem.text.strip()
                    
                    if match_data:
                        live_matches.append(match_data)
                        
                except Exception as e:
                    logger.debug(f"Error parsing live match: {e}")
                    continue
            
            return live_matches
            
        except Exception as e:
            logger.error(f"Error fetching live matches: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
        
        if self.executor:
            self.executor.shutdown(wait=True)


# Example usage and testing
async def main():
    """Example usage of HLTV Stats Scraper"""
    scraper = HLTVStatsScraper()
    
    try:
        await scraper.initialize()
        
        # Test team stats
        team_stats = await scraper.get_team_stats("NAVI")
        print(f"Team stats for NAVI: {json.dumps(team_stats, indent=2)}")
        
        # Test live matches
        live_matches = await scraper.get_live_matches()
        print(f"Live matches: {json.dumps(live_matches, indent=2)}")
        
    finally:
        await scraper.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
