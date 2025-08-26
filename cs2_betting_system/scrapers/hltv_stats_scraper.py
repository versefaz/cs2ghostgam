import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

# Optional heavy deps guarded
try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    import cloudscraper  # type: ignore
except Exception:  # pragma: no cover
    cloudscraper = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    import undetected_chromedriver as uc  # type: ignore
    from selenium.webdriver.common.by import By  # type: ignore
    from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
    from selenium.webdriver.support import expected_conditions as EC  # type: ignore
    SELENIUM_AVAILABLE = True
except Exception:  # pragma: no cover
    SELENIUM_AVAILABLE = False

try:
    import redis  # type: ignore
    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HLTVStatsScraper:
    """
    ระบบดึงข้อมูลสถิติจาก HLTV แบบ High-Performance
    ใช้ multi-method scraping และ caching strategy
    """

    def __init__(self):
        self.base_url = "https://www.hltv.org"
        self.scraper = cloudscraper.create_scraper() if cloudscraper else None
        self.session: Optional[object] = None
        self.driver = None
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Cache setup: prefer Redis, fallback to in-memory
        if REDIS_AVAILABLE:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                self.cache.ping()
            except Exception:
                self.cache = None
        else:
            self.cache = None
        self._mem_cache: Dict[str, Dict] = {}

        # Rate limiting
        self.last_request: Dict[str, float] = {}
        self.min_delay = 1.0  # seconds between requests per domain

        # Headers สำหรับ bypass detection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    async def initialize(self):
        """Initialize async session และ selenium driver"""
        if aiohttp:
            self.session = aiohttp.ClientSession(headers=self.headers)

        # Setup undetected Chrome driver
        if SELENIUM_AVAILABLE:
            options = uc.ChromeOptions()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            try:
                self.driver = uc.Chrome(options=options)
                logger.info("Selenium driver initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Selenium: {e}")

    def _cache_get(self, key: str) -> Optional[Dict]:
        try:
            if self.cache:
                v = self.cache.get(key)
                return json.loads(v) if v else None
        except Exception:
            pass
        return self._mem_cache.get(key)

    def _cache_setex(self, key: str, ttl: int, value: Dict) -> None:
        try:
            if self.cache:
                self.cache.setex(key, ttl, json.dumps(value))
                return
        except Exception:
            pass
        self._mem_cache[key] = value

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

        if not force_refresh:
            cached = self._cache_get(cache_key)
            if cached:
                logger.info(f"Using cached data for {team_name}")
                return cached

        logger.info(f"Fetching fresh data for {team_name}")

        team_id = await self._get_team_id(team_name)
        if not team_id:
            logger.error(f"Team {team_name} not found")
            return {}

        tasks = [
            self._fetch_team_overview(team_id),
            self._fetch_team_matches(team_id, days=90),
            self._fetch_map_stats(team_id),
            self._fetch_player_stats(team_id),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        team_stats = {
            'team_name': team_name,
            'team_id': team_id,
            'last_updated': datetime.utcnow().isoformat(),
            'overview': results[0] if not isinstance(results[0], Exception) else {},
            'recent_matches': results[1] if not isinstance(results[1], Exception) else [],
            'map_stats': results[2] if not isinstance(results[2], Exception) else {},
            'player_stats': results[3] if not isinstance(results[3], Exception) else [],
        }

        team_stats['processed'] = self._process_team_metrics(team_stats)

        self._cache_setex(cache_key, 3600, team_stats)
        return team_stats

    async def _get_team_id(self, team_name: str) -> Optional[int]:
        """ค้นหา team ID จากชื่อทีม"""
        search_url = f"{self.base_url}/search?query={team_name}"
        try:
            response = await self._fetch_with_retry(search_url)
            if not response or not BeautifulSoup:
                return None
            soup = BeautifulSoup(response.text, 'html.parser')
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
        url = f"{self.base_url}/team/{team_id}/"
        try:
            response = await self._fetch_with_retry(url)
            if not response or not BeautifulSoup:
                return {}
            soup = BeautifulSoup(response.text, 'html.parser')
            overview: Dict = {}

            ranking = soup.find('div', class_='profile-team-stat')
            if ranking:
                rank_text = ranking.find('span', class_='value')
                if rank_text:
                    m = re.search(r'\d+', rank_text.text)
                    if m:
                        overview['world_ranking'] = int(m.group())

            team_info = soup.find('div', class_='profile-team-info')
            if team_info:
                flag = team_info.find('img', class_='flag')
                overview['country'] = flag.get('title') if flag else 'Unknown'

            stats_container = soup.find('div', class_='profile-team-stats-container')
            if stats_container:
                stat_values = stats_container.find_all('div', class_='stat-value')
                try:
                    if len(stat_values) >= 3:
                        m0 = re.search(r'[\d.]+', stat_values[0].text or '')
                        m1 = re.search(r'\d+', stat_values[1].text or '')
                        m2 = re.search(r'\d+', stat_values[2].text or '')
                        if m0:
                            overview['avg_player_age'] = float(m0.group())
                        if m1:
                            overview['weeks_in_top30'] = int(m1.group())
                        if m2:
                            overview['trophies'] = int(m2.group())
                except Exception:
                    pass

            lineup = []
            players_container = soup.find('div', class_='bodyshot-team')
            if players_container:
                players = players_container.find_all('a', class_='player-link')
                for player in players:
                    name_node = player.find('span', class_='text-ellipsis')
                    if name_node and name_node.text:
                        lineup.append(name_node.text.strip())
            overview['current_lineup'] = lineup

            return overview
        except Exception as e:
            logger.error(f"Error fetching team overview: {e}")
            return {}

    async def _fetch_team_matches(self, team_id: int, days: int = 90) -> List[Dict]:
        url = f"{self.base_url}/results?team={team_id}"
        try:
            response = await self._fetch_with_retry(url)
            if not response or not BeautifulSoup:
                return []
            soup = BeautifulSoup(response.text, 'html.parser')
            matches: List[Dict] = []
            match_elements = soup.find_all('div', class_='result-con', limit=50)
            for match_elem in match_elements:
                try:
                    match_data: Dict = {}
                    date_elem = match_elem.find('div', class_='date')
                    if date_elem:
                        match_data['date'] = date_elem.text.strip()
                    team_elems = match_elem.find_all('div', class_='team')
                    score_elem = match_elem.find('span', class_='score')
                    if len(team_elems) >= 2 and score_elem:
                        scores = score_elem.text.strip().split('-')
                        if len(scores) == 2:
                            team1_name = team_elems[0].text.strip()
                            team2_name = team_elems[1].text.strip()
                            # Simplified: assume our team is first when present
                            href_a = match_elem.find('a')
                            our_team_first = bool(href_a and str(team_id) in href_a.get('href', ''))
                            match_data['opponent'] = team2_name if our_team_first else team1_name
                            match_data['score'] = f"{scores[0]}-{scores[1]}"
                            match_data['our_score'] = int(re.sub(r'\D', '', scores[0] if our_team_first else scores[1]) or 0)
                            match_data['opp_score'] = int(re.sub(r'\D', '', scores[1] if our_team_first else scores[0]) or 0)
                            match_data['result'] = 'W' if match_data['our_score'] > match_data['opp_score'] else 'L'
                    map_elem = match_elem.find('div', class_='map')
                    if map_elem:
                        match_data['map'] = map_elem.text.strip().lower()
                    event_elem = match_elem.find('div', class_='event')
                    if event_elem:
                        match_data['event'] = event_elem.text.strip()
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.warning(f"Error parsing match: {e}")
                    continue
            return matches
        except Exception as e:
            logger.error(f"Error fetching team matches: {e}")
            return []

    async def _fetch_map_stats(self, team_id: int) -> Dict:
        url = f"{self.base_url}/stats/teams/maps/{team_id}/"
        try:
            if self.driver:
                self.driver.get(url)
                await asyncio.sleep(2)
                if not BeautifulSoup:
                    return {}
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            else:
                response = await self._fetch_with_retry(url)
                if not response or not BeautifulSoup:
                    return {}
                soup = BeautifulSoup(response.text, 'html.parser')

            map_stats: Dict[str, Dict] = {}
            stats_table = soup.find('table', class_='stats-table')
            if stats_table:
                rows = stats_table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 5:
                        map_name = (cols[0].text or '').strip().lower()
                        try:
                            mp = int(re.sub(r'\D', '', cols[1].text))
                            w = int(re.sub(r'\D', '', cols[2].text))
                            d = int(re.sub(r'\D', '', cols[3].text))
                            l = int(re.sub(r'\D', '', cols[4].text))
                            wr = float((cols[5].text or '0').replace('%', '').strip()) / 100 if len(cols) > 5 else 0.0
                            rw = int(re.sub(r'\D', '', cols[6].text)) if len(cols) > 6 else 0
                            rl = int(re.sub(r'\D', '', cols[7].text)) if len(cols) > 7 else 0
                        except Exception:
                            continue
                        item = {
                            'matches_played': mp,
                            'wins': w,
                            'draws': d,
                            'losses': l,
                            'win_rate': wr,
                            'rounds_won': rw,
                            'rounds_lost': rl,
                        }
                        total = w + l
                        if total > 0:
                            item['calculated_winrate'] = w / total
                        tr = rw + rl
                        if tr > 0:
                            item['round_winrate'] = rw / tr
                        map_stats[map_name] = item

            if not map_stats:
                matches = await self._fetch_team_matches(team_id)
                map_stats = self._calculate_map_stats_from_matches(matches)
            return map_stats
        except Exception as e:
            logger.error(f"Error fetching map stats: {e}")
            return {}

    async def _fetch_player_stats(self, team_id: int) -> List[Dict]:
        url = f"{self.base_url}/stats/teams/players/{team_id}/"
        try:
            response = await self._fetch_with_retry(url)
            if not response or not BeautifulSoup:
                return []
            soup = BeautifulSoup(response.text, 'html.parser')
            players: List[Dict] = []
            stats_table = soup.find('table', class_='stats-table')
            if stats_table:
                rows = stats_table.find_all('tr')[1:]
                for row in rows[:5]:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        name_node = cols[0].find('a') or cols[0]
                        name = (name_node.text or '').strip()
                        try:
                            players.append({
                                'name': name,
                                'maps_played': int(re.sub(r'\D', '', cols[1].text)),
                                'kd_diff': float((cols[2].text or '0').replace('+', '').strip()),
                                'kd_ratio': float(cols[3].text or 0),
                                'rating_2': float(cols[4].text or 1.0),
                                'adr': float(cols[5].text or 0) if len(cols) > 5 else 0.0,
                                'kast': float((cols[6].text or '0').replace('%', '').strip()) / 100 if len(cols) > 6 else 0.0,
                            })
                        except Exception:
                            continue
            return players
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            return []

    async def get_h2h_stats(self, team1_name: str, team2_name: str) -> Dict:
        cache_key = f"hltv:h2h:{team1_name}:{team2_name}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        team1_id = await self._get_team_id(team1_name)
        team2_id = await self._get_team_id(team2_name)
        if not team1_id or not team2_id:
            logger.error("Could not find team IDs for H2H")
            return {}

        url = f"{self.base_url}/results?team={team1_id}&team={team2_id}"
        try:
            response = await self._fetch_with_retry(url)
            if not response or not BeautifulSoup:
                return {}
            soup = BeautifulSoup(response.text, 'html.parser')
            h2h_matches: List[Dict] = []
            match_elements = soup.find_all('div', class_='result-con', limit=20)
            for match_elem in match_elements:
                try:
                    md: Dict = {}
                    date_elem = match_elem.find('div', class_='date')
                    if date_elem:
                        md['date'] = date_elem.text.strip()
                    score_elem = match_elem.find('span', class_='score')
                    if score_elem:
                        md['score'] = score_elem.text.strip()
                    map_elem = match_elem.find('div', class_='map')
                    if map_elem:
                        md['map'] = map_elem.text.strip().lower()
                    team_elems = match_elem.find_all('div', class_='team')
                    if len(team_elems) >= 2:
                        if 'team-won' in (team_elems[0].get('class') or []):
                            md['winner'] = team_elems[0].text.strip()
                        elif 'team-won' in (team_elems[1].get('class') or []):
                            md['winner'] = team_elems[1].text.strip()
                    h2h_matches.append(md)
                except Exception as e:
                    logger.warning(f"Error parsing H2H match: {e}")
                    continue

            t1w = sum(1 for m in h2h_matches if team1_name.lower() in (m.get('winner', '').lower()))
            t2w = sum(1 for m in h2h_matches if team2_name.lower() in (m.get('winner', '').lower()))
            h2h_stats = {
                'total_matches': len(h2h_matches),
                'team1_wins': t1w,
                'team2_wins': t2w,
                'team1_winrate': (t1w / len(h2h_matches)) if h2h_matches else 0.0,
                'recent_matches': h2h_matches[:5],
                'map_breakdown': self._calculate_map_h2h(h2h_matches),
            }
            self._cache_setex(cache_key, 7200, h2h_stats)
            return h2h_stats
        except Exception as e:
            logger.error(f"Error fetching H2H stats: {e}")
            return {}

    def _process_team_metrics(self, team_stats: Dict) -> Dict:
        processed: Dict = {}  # Fix syntax error by replacing stray placeholder with proper initialization
        recent_matches = team_stats.get('recent_matches', [])
        if recent_matches:
            last_10 = recent_matches[:10]
            wins = sum(1 for m in last_10 if m.get('result') == 'W')
            processed['form_last_10'] = wins / len(last_10)
            last_5 = recent_matches[:5]
            wins_5 = sum(1 for m in last_5 if m.get('result') == 'W')
            processed['form_last_5'] = wins_5 / len(last_5)
            weights = [1.0, 0.9, 0.8, 0.7, 0.6]
            weighted_wins = sum(w * (1 if recent_matches[i].get('result') == 'W' else 0) for i, w in enumerate(weights) if i < len(recent_matches))
            processed['momentum'] = weighted_wins / sum(weights[:len(recent_matches[:5])])
            round_diffs = [m.get('our_score', 0) - m.get('opp_score', 0) for m in last_10]
            if round_diffs:
                processed['avg_round_diff'] = sum(round_diffs) / len(round_diffs)
        map_stats = team_stats.get('map_stats', {})
        if map_stats:
            active_maps = ['ancient', 'anubis', 'inferno', 'mirage', 'nuke', 'overpass', 'vertigo']
            map_strengths: Dict[str, Dict] = {}
            for map_name in active_maps:
                if map_name in map_stats:
                    stats = map_stats[map_name]
                    play_weight = min(stats.get('matches_played', 0) / 20, 1.0)
                    strength = stats.get('win_rate', 0.5) * play_weight
                    map_strengths[map_name] = {'strength': strength, 'winrate': stats.get('win_rate', 0.5), 'matches': stats.get('matches_played', 0)}
            processed['map_pool'] = map_strengths
            processed['avg_map_strength'] = sum(m['strength'] for m in map_strengths.values()) / len(map_strengths) if map_strengths else 0
        player_stats = team_stats.get('player_stats', [])
        if player_stats:
            ratings = [p.get('rating_2', 1.0) for p in player_stats]
            if ratings:
                processed['avg_team_rating'] = sum(ratings) / len(ratings)
                processed['star_player_rating'] = max(ratings)
                if len(ratings) > 1:
                    # simple consistency proxy
                    mean = processed['avg_team_rating']
                    var = sum((r - mean) ** 2 for r in ratings) / (len(ratings) - 1)
                    std = var ** 0.5
                    processed['consistency'] = max(0.0, 1 - std)
                else:
                    processed['consistency'] = 0.0
        strength_factors: List[float] = []
        ranking = team_stats.get('overview', {}).get('world_ranking', 30)
        ranking_score = max(0, (30 - ranking) / 30)
        strength_factors.append(ranking_score * 0.3)
        if 'form_last_10' in processed:
            strength_factors.append(processed['form_last_10'] * 0.3)
        if 'avg_team_rating' in processed:
            rating_normalized = (processed['avg_team_rating'] - 0.9) / 0.4
            rating_normalized = min(max(rating_normalized, 0), 1)
            strength_factors.append(rating_normalized * 0.2)
        if 'avg_map_strength' in processed:
            strength_factors.append(processed['avg_map_strength'] * 0.2)
        if strength_factors:
            processed['overall_strength'] = sum(strength_factors)
        return processed

    def _calculate_map_stats_from_matches(self, matches: List[Dict]) -> Dict:
        map_stats: Dict[str, Dict] = {}
        for match in matches:
            map_name = (match.get('map') or '').lower()
            if not map_name or map_name == 'tba':
                continue
            ms = map_stats.setdefault(map_name, {'matches_played': 0, 'wins': 0, 'losses': 0, 'rounds_won': 0, 'rounds_lost': 0})
            ms['matches_played'] += 1
            if match.get('result') == 'W':
                ms['wins'] += 1
            else:
                ms['losses'] += 1
            ms['rounds_won'] += int(match.get('our_score', 0))
            ms['rounds_lost'] += int(match.get('opp_score', 0))
        for map_name, stats in map_stats.items():
            total = stats['wins'] + stats['losses']
            if total > 0:
                stats['win_rate'] = stats['wins'] / total
            total_rounds = stats['rounds_won'] + stats['rounds_lost']
            if total_rounds > 0:
                stats['round_winrate'] = stats['rounds_won'] / total_rounds
        return map_stats

    def _calculate_map_h2h(self, h2h_matches: List[Dict]) -> Dict:
        map_h2h: Dict[str, Dict] = {}
        for match in h2h_matches:
            map_name = (match.get('map') or '').lower()
            if not map_name or map_name == 'tba':
                continue
            mh = map_h2h.setdefault(map_name, {'matches': 0, 'team1_wins': 0, 'team2_wins': 0})
            mh['matches'] += 1
        return map_h2h

    async def _fetch_with_retry(self, url: str, max_retries: int = 3):
        domain = url.split('/')[2]
        if domain in self.last_request:
            elapsed = time.time() - self.last_request[domain]
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
        for attempt in range(max_retries):
            try:
                if self.scraper:
                    response = self.scraper.get(url, headers=self.headers)
                    if getattr(response, 'status_code', None) == 200:
                        self.last_request[domain] = time.time()
                        return response
                if self.session and aiohttp:
                    async with self.session.get(url) as resp:
                        if resp.status == 200:
                            self.last_request[domain] = time.time()
                            text = await resp.text()
                            return type('Response', (), {'text': text, 'url': str(resp.url)})()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        raise Exception(f"Failed to fetch {url} after {max_retries} attempts")

    async def close(self):
        if self.session and aiohttp:
            await self.session.close()
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
        self.executor.shutdown(wait=True)
