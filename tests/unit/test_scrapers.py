import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json

from cs2_betting_system.scrapers.live_match_scraper import LiveMatchScraper
from cs2_betting_system.scrapers.hltv_stats_scraper import HLTVStatsScraper
from cs2_betting_system.scrapers.enhanced_live_match_scraper import EnhancedLiveMatchScraper


class TestLiveMatchScraper:
    """Unit tests for LiveMatchScraper"""
    
    @pytest.fixture
    def scraper(self):
        """Create LiveMatchScraper instance for testing"""
        return LiveMatchScraper()
    
    @pytest.fixture
    def mock_driver(self):
        """Mock Selenium WebDriver"""
        driver = Mock()
        driver.get = Mock()
        driver.find_elements.return_value = []
        driver.page_source = "<html><body>Mock page</body></html>"
        driver.quit = Mock()
        return driver
    
    def test_scraper_initialization(self, scraper):
        """Test scraper initialization"""
        assert scraper.base_url == "https://www.hltv.org"
        assert scraper.driver is None
        assert isinstance(scraper.matches_cache, dict)
    
    @patch('cs2_betting_system.scrapers.live_match_scraper.webdriver.Chrome')
    def test_setup_driver(self, mock_chrome, scraper):
        """Test WebDriver setup"""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        scraper.setup_driver()
        
        assert scraper.driver == mock_driver
        mock_chrome.assert_called_once()
    
    def test_cleanup(self, scraper, mock_driver):
        """Test cleanup functionality"""
        scraper.driver = mock_driver
        
        scraper.cleanup()
        
        mock_driver.quit.assert_called_once()
        assert scraper.driver is None
    
    @patch('cs2_betting_system.scrapers.live_match_scraper.redis.Redis')
    def test_get_cached_odds(self, mock_redis, scraper):
        """Test cached odds retrieval"""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        
        mock_odds_data = {
            'odds_team1': 1.85,
            'odds_team2': 1.95,
            'odds_source': 'test_source',
            'timestamp': datetime.now().isoformat()
        }
        mock_redis_client.get.return_value = json.dumps(mock_odds_data)
        
        result = scraper.get_cached_odds('Team A', 'Team B')
        
        assert result == mock_odds_data
        mock_redis_client.get.assert_called_once()
    
    def test_parse_match_element(self, scraper):
        """Test match element parsing"""
        mock_element = Mock()
        mock_element.find_elements.return_value = [
            Mock(text='Team A'),
            Mock(text='Team B')
        ]
        mock_element.find_element.return_value = Mock(text='16:30')
        
        with patch.object(scraper, '_extract_team_names', return_value=['Team A', 'Team B']):
            with patch.object(scraper, '_extract_match_time', return_value='16:30'):
                result = scraper._parse_match_element(mock_element)
                
                assert isinstance(result, dict)
                assert result['team1'] == 'Team A'
                assert result['team2'] == 'Team B'
                assert 'match_id' in result
    
    def test_scrape_all_sources_no_driver(self, scraper):
        """Test scraping when no driver is available"""
        scraper.driver = None
        
        result = scraper.scrape_all_sources()
        
        assert isinstance(result, list)
        assert len(result) == 0  # Should return empty list when no driver
    
    @patch('cs2_betting_system.scrapers.live_match_scraper.webdriver.Chrome')
    def test_scrape_all_sources_with_driver(self, mock_chrome, scraper):
        """Test scraping with driver available"""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        mock_driver.find_elements.return_value = []
        
        scraper.setup_driver()
        result = scraper.scrape_all_sources()
        
        assert isinstance(result, list)
        mock_driver.get.assert_called()


class TestHLTVStatsScraper:
    """Unit tests for HLTVStatsScraper"""
    
    @pytest.fixture
    def stats_scraper(self):
        """Create HLTVStatsScraper instance for testing"""
        return HLTVStatsScraper()
    
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session"""
        session = AsyncMock()
        response = AsyncMock()
        response.text = "<html><body>Mock HLTV page</body></html>"
        response.status = 200
        session.get.return_value.__aenter__.return_value = response
        return session
    
    def test_stats_scraper_initialization(self, stats_scraper):
        """Test stats scraper initialization"""
        assert stats_scraper.base_url == "https://www.hltv.org"
        assert stats_scraper.session is None
        assert isinstance(stats_scraper._mem_cache, dict)
    
    @pytest.mark.asyncio
    async def test_initialize(self, stats_scraper):
        """Test async initialization"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await stats_scraper.initialize()
            
            assert stats_scraper.session == mock_session
    
    def test_cache_operations(self, stats_scraper):
        """Test cache get/set operations"""
        test_key = "test_key"
        test_value = {"test": "data"}
        
        # Test cache miss
        result = stats_scraper._cache_get(test_key)
        assert result is None
        
        # Test cache set and get
        stats_scraper._cache_setex(test_key, 3600, test_value)
        result = stats_scraper._cache_get(test_key)
        assert result == test_value
    
    @pytest.mark.asyncio
    async def test_get_team_id(self, stats_scraper):
        """Test team ID retrieval"""
        mock_html = '''
        <html>
            <body>
                <a href="/team/4608/natus-vincere">Natus Vincere</a>
            </body>
        </html>
        '''
        
        with patch.object(stats_scraper, '_fetch_with_retry') as mock_fetch:
            mock_response = Mock()
            mock_response.text = mock_html
            mock_fetch.return_value = mock_response
            
            team_id = await stats_scraper._get_team_id("Natus Vincere")
            
            assert team_id == 4608
    
    @pytest.mark.asyncio
    async def test_get_team_stats_cached(self, stats_scraper):
        """Test team stats retrieval from cache"""
        cached_data = {
            'team_name': 'Test Team',
            'last_updated': datetime.now().isoformat()
        }
        
        with patch.object(stats_scraper, '_cache_get', return_value=cached_data):
            result = await stats_scraper.get_team_stats('Test Team')
            
            assert result == cached_data
    
    @pytest.mark.asyncio
    async def test_get_team_stats_fresh(self, stats_scraper):
        """Test fresh team stats retrieval"""
        with patch.object(stats_scraper, '_cache_get', return_value=None):
            with patch.object(stats_scraper, '_get_team_id', return_value=1234):
                with patch.object(stats_scraper, '_fetch_team_overview', return_value={}):
                    with patch.object(stats_scraper, '_fetch_team_matches', return_value=[]):
                        with patch.object(stats_scraper, '_fetch_map_stats', return_value={}):
                            with patch.object(stats_scraper, '_fetch_player_stats', return_value=[]):
                                with patch.object(stats_scraper, '_cache_setex'):
                                    result = await stats_scraper.get_team_stats('Test Team')
                                    
                                    assert isinstance(result, dict)
                                    assert result['team_name'] == 'Test Team'
                                    assert result['team_id'] == 1234
                                    assert 'last_updated' in result
    
    @pytest.mark.asyncio
    async def test_get_h2h_stats(self, stats_scraper):
        """Test head-to-head stats retrieval"""
        with patch.object(stats_scraper, '_cache_get', return_value=None):
            with patch.object(stats_scraper, '_get_team_id', side_effect=[1234, 5678]):
                mock_html = '''
                <html>
                    <body>
                        <div class="result-con">
                            <div class="date">2024-01-15</div>
                            <span class="score">16-12</span>
                            <div class="map">mirage</div>
                            <div class="team team-won">Team A</div>
                            <div class="team">Team B</div>
                        </div>
                    </body>
                </html>
                '''
                
                with patch.object(stats_scraper, '_fetch_with_retry') as mock_fetch:
                    mock_response = Mock()
                    mock_response.text = mock_html
                    mock_fetch.return_value = mock_response
                    
                    with patch.object(stats_scraper, '_cache_setex'):
                        result = await stats_scraper.get_h2h_stats('Team A', 'Team B')
                        
                        assert isinstance(result, dict)
                        assert 'total_matches' in result
                        assert 'team1_wins' in result
                        assert 'team2_wins' in result
    
    def test_process_team_metrics(self, stats_scraper):
        """Test team metrics processing"""
        team_stats = {
            'recent_matches': [
                {'result': 'W', 'our_score': 16, 'opp_score': 12},
                {'result': 'W', 'our_score': 16, 'opp_score': 8},
                {'result': 'L', 'our_score': 12, 'opp_score': 16},
                {'result': 'W', 'our_score': 16, 'opp_score': 14}
            ],
            'map_stats': {
                'mirage': {'win_rate': 0.75, 'matches_played': 20},
                'inferno': {'win_rate': 0.68, 'matches_played': 15}
            },
            'player_stats': [
                {'rating_2': 1.25},
                {'rating_2': 1.18},
                {'rating_2': 1.05}
            ],
            'overview': {'world_ranking': 5}
        }
        
        processed = stats_scraper._process_team_metrics(team_stats)
        
        assert isinstance(processed, dict)
        assert 'form_last_10' in processed
        assert 'avg_team_rating' in processed
        assert 'overall_strength' in processed
        
        # Check form calculation (3 wins out of 4 matches)
        assert processed['form_last_10'] == 0.75
        
        # Check average rating
        expected_avg = (1.25 + 1.18 + 1.05) / 3
        assert abs(processed['avg_team_rating'] - expected_avg) < 0.01
    
    @pytest.mark.asyncio
    async def test_fetch_with_retry_success(self, stats_scraper):
        """Test successful fetch with retry"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Success"
        
        with patch.object(stats_scraper, 'scraper') as mock_scraper:
            mock_scraper.get.return_value = mock_response
            
            result = await stats_scraper._fetch_with_retry("http://test.com")
            
            assert result == mock_response
            mock_scraper.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_with_retry_failure(self, stats_scraper):
        """Test fetch with retry on failure"""
        with patch.object(stats_scraper, 'scraper') as mock_scraper:
            mock_scraper.get.side_effect = Exception("Network error")
            
            with pytest.raises(Exception):
                await stats_scraper._fetch_with_retry("http://test.com", max_retries=2)
            
            assert mock_scraper.get.call_count == 2


class TestEnhancedLiveMatchScraper:
    """Unit tests for EnhancedLiveMatchScraper"""
    
    @pytest.fixture
    def enhanced_scraper(self):
        """Create EnhancedLiveMatchScraper instance for testing"""
        return EnhancedLiveMatchScraper()
    
    @pytest.mark.asyncio
    async def test_initialize(self, enhanced_scraper):
        """Test enhanced scraper initialization"""
        with patch.object(enhanced_scraper.hltv_scraper, 'initialize') as mock_hltv_init:
            with patch.object(enhanced_scraper.prediction_model, 'initialize') as mock_model_init:
                await enhanced_scraper.initialize()
                
                mock_hltv_init.assert_called_once()
                mock_model_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_live_matches(self, enhanced_scraper):
        """Test live matches retrieval"""
        mock_matches = [
            {
                'team_a': 'Team A',
                'team_b': 'Team B',
                'status': 'live',
                'map': 'mirage'
            }
        ]
        
        def mock_scrape():
            scraper = Mock()
            scraper.scrape_all_sources.return_value = mock_matches
            scraper.cleanup = Mock()
            return mock_matches
        
        with patch('asyncio.get_running_loop') as mock_loop:
            mock_loop.return_value.run_in_executor.return_value = mock_matches
            
            result = await enhanced_scraper.get_live_matches()
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]['team1'] == 'Team A'
            assert result[0]['team2'] == 'Team B'
    
    @pytest.mark.asyncio
    async def test_enhance_match_with_stats(self, enhanced_scraper, sample_match_data):
        """Test match enhancement with stats"""
        mock_team_stats = {'processed': {'form_last_10': 0.8}}
        mock_h2h_stats = {'total_matches': 5}
        mock_odds = {'average': {'team1': 1.85, 'team2': 1.95}}
        
        with patch.object(enhanced_scraper.hltv_scraper, 'get_team_stats', return_value=mock_team_stats):
            with patch.object(enhanced_scraper.hltv_scraper, 'get_h2h_stats', return_value=mock_h2h_stats):
                with patch.object(enhanced_scraper, '_fetch_live_odds', return_value=mock_odds):
                    result = await enhanced_scraper._enhance_match_with_stats(sample_match_data)
                    
                    assert isinstance(result, dict)
                    assert 'team1_stats' in result
                    assert 'team2_stats' in result
                    assert 'h2h_stats' in result
                    assert 'odds' in result
                    assert result['features_ready'] is True
    
    @pytest.mark.asyncio
    async def test_fetch_live_odds_multi_source(self, enhanced_scraper, sample_match_data):
        """Test live odds fetching from multiple sources"""
        mock_odds_results = [
            {'team1_odds': 1.85, 'team2_odds': 1.95, 'source': 'bet365'},
            {'team1_odds': 1.88, 'team2_odds': 1.92, 'source': 'pinnacle'},
            None  # Failed source
        ]
        
        with patch.object(enhanced_scraper, '_get_odds_from_source', side_effect=mock_odds_results):
            result = await enhanced_scraper._fetch_live_odds(sample_match_data)
            
            assert isinstance(result, dict)
            assert 'sources' in result
            assert 'average' in result
            assert 'best' in result
            
            # Should have 2 successful sources
            assert len(result['sources']) == 2
            
            # Check averages
            assert abs(result['average']['team1'] - 1.865) < 0.01  # (1.85 + 1.88) / 2
            assert abs(result['average']['team2'] - 1.935) < 0.01  # (1.95 + 1.92) / 2
    
    @pytest.mark.asyncio
    async def test_get_odds_from_source_success(self, enhanced_scraper, sample_match_data):
        """Test successful odds retrieval from source"""
        mock_odds = {
            'odds_team1': 1.85,
            'odds_team2': 1.95,
            'odds_source': 'test_source'
        }
        
        def mock_fetch():
            scraper = Mock()
            scraper.get_cached_odds.return_value = mock_odds
            scraper.cleanup = Mock()
            return mock_odds
        
        with patch('asyncio.get_running_loop') as mock_loop:
            mock_loop.return_value.run_in_executor.return_value = mock_odds
            
            result = await enhanced_scraper._get_odds_from_source(sample_match_data, 'test_source')
            
            assert isinstance(result, dict)
            assert result['team1_odds'] == 1.85
            assert result['team2_odds'] == 1.95
            assert result['source'] == 'test_source'
    
    @pytest.mark.asyncio
    async def test_get_odds_from_source_failure(self, enhanced_scraper, sample_match_data):
        """Test odds retrieval failure"""
        with patch('asyncio.get_running_loop') as mock_loop:
            mock_loop.return_value.run_in_executor.side_effect = Exception("Network error")
            
            result = await enhanced_scraper._get_odds_from_source(sample_match_data, 'test_source')
            
            assert result is None
    
    def test_detect_odds_movement(self, enhanced_scraper, sample_match_data):
        """Test odds movement detection"""
        current_odds = {
            'average': {'team1': 1.90, 'team2': 1.90},
            'timestamp': datetime.now().isoformat()
        }
        
        # Set up historical data
        enhanced_scraper.matches_cache[sample_match_data['match_id']] = {
            'odds_history': [
                {
                    'average': {'team1': 1.80, 'team2': 2.00},
                    'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat()
                }
            ]
        }
        
        result = asyncio.run(enhanced_scraper._detect_odds_movement(sample_match_data, current_odds))
        
        assert isinstance(result, dict)
        assert 'direction' in result
        assert 'magnitude' in result
        assert 'velocity' in result
        
        # Should detect upward movement for team1 (1.80 -> 1.90)
        assert result['direction'] == 'up'
        assert result['magnitude'] == 0.10
