import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import redis
from fastapi.testclient import TestClient

from shared.redis_schema import RedisSchemaManager, KeyType
from services.api_gateway.app.api_endpoints import app
from tests.fixtures.match_fixtures import MockRedisClient


class TestTrackerRedisAPIIntegration:
    """Integration tests for tracker → Redis → API pipeline"""
    
    @pytest.fixture
    def redis_client(self, mock_redis_data):
        """Mock Redis client for integration testing"""
        return MockRedisClient(mock_redis_data)
    
    @pytest.fixture
    def schema_manager(self, redis_client):
        """Redis schema manager with mock client"""
        return RedisSchemaManager(redis_client)
    
    @pytest.fixture
    def api_client(self, redis_client):
        """FastAPI test client with mocked Redis"""
        with patch('services.api_gateway.app.api_endpoints.redis_client', redis_client):
            with patch('services.api_gateway.app.api_endpoints.schema_manager') as mock_schema:
                mock_schema.return_value = RedisSchemaManager(redis_client)
                yield TestClient(app)
    
    @pytest.mark.asyncio
    async def test_prediction_flow_end_to_end(self, schema_manager, api_client, enhanced_match_with_stats, sample_prediction_result):
        """Test complete prediction flow from tracker to API"""
        
        # Step 1: Simulate tracker storing prediction in Redis
        match_id = enhanced_match_with_stats['match_id']
        
        # Store prediction
        success = schema_manager.set_with_ttl(
            KeyType.PREDICTION, 'current',
            sample_prediction_result,
            match_id=match_id
        )
        assert success
        
        # Store in recent predictions list
        schema_manager.redis.lpush(
            'predictions:recent',
            json.dumps(sample_prediction_result, default=str)
        )
        
        # Step 2: Test API retrieval
        response = api_client.get(f"/predictions/{match_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data['prediction']['match_id'] == match_id
        assert data['prediction']['team1'] == 'Natus Vincere'
        assert data['prediction']['team2'] == 'FaZe Clan'
        assert 'expected_value' in data['prediction']
        assert 'recommendation' in data['prediction']
        
        # Step 3: Test recent predictions endpoint
        response = api_client.get("/predictions/recent?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert data['count'] >= 1
        assert len(data['predictions']) >= 1
        
        # Find our prediction in recent list
        our_prediction = next(
            (p for p in data['predictions'] if p['match_id'] == match_id),
            None
        )
        assert our_prediction is not None
    
    @pytest.mark.asyncio
    async def test_signal_processing_flow(self, schema_manager, api_client, sample_signal_data):
        """Test signal processing from generation to API"""
        
        # Step 1: Store active signal
        signal_id = sample_signal_data['id']
        success = schema_manager.set_with_ttl(
            KeyType.SIGNAL, 'active',
            sample_signal_data,
            signal_id=signal_id
        )
        assert success
        
        # Step 2: Add to processing queue
        queue_item = {
            'signal_id': signal_id,
            'priority': 'high',
            'timestamp': datetime.now().isoformat()
        }
        schema_manager.redis.lpush('queue:high_priority', json.dumps(queue_item))
        
        # Step 3: Test API endpoints
        
        # Get active signals
        response = api_client.get("/signals/active")
        assert response.status_code == 200
        
        data = response.json()
        assert data['count'] >= 1
        
        # Find our signal
        our_signal = next(
            (s for s in data['signals'] if s['id'] == signal_id),
            None
        )
        assert our_signal is not None
        assert our_signal['team'] == 'Natus Vincere'
        assert our_signal['confidence'] == 78
        
        # Get queue status
        response = api_client.get("/signals/queue")
        assert response.status_code == 200
        
        data = response.json()
        assert 'queue_stats' in data
        assert 'high_priority' in data['queue_stats']
        assert data['queue_stats']['high_priority']['length'] >= 1
    
    @pytest.mark.asyncio
    async def test_match_data_flow(self, schema_manager, api_client, enhanced_match_with_stats):
        """Test match data flow from scraper to API"""
        
        match_id = enhanced_match_with_stats['match_id']
        
        # Step 1: Store live match data (simulating scraper)
        success = schema_manager.set_with_ttl(
            KeyType.MATCH, 'live',
            enhanced_match_with_stats,
            match_id=match_id
        )
        assert success
        
        # Step 2: Store upcoming match
        upcoming_match = enhanced_match_with_stats.copy()
        upcoming_match['status'] = 'upcoming'
        upcoming_match['start_time'] = (datetime.now() + timedelta(hours=2)).isoformat()
        
        success = schema_manager.set_with_ttl(
            KeyType.MATCH, 'upcoming',
            upcoming_match,
            match_id=f"{match_id}_upcoming"
        )
        assert success
        
        # Step 3: Test API endpoints
        
        # Get live matches
        response = api_client.get("/matches/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data['count'] >= 1
        
        live_match = next(
            (m for m in data['matches'] if m['match_id'] == match_id),
            None
        )
        assert live_match is not None
        assert live_match['status'] == 'live'
        
        # Get upcoming matches
        response = api_client.get("/matches/upcoming?hours=24")
        assert response.status_code == 200
        
        data = response.json()
        assert data['count'] >= 1
        assert data['hours_ahead'] == 24
    
    @pytest.mark.asyncio
    async def test_odds_data_flow(self, schema_manager, api_client, sample_odds_data):
        """Test odds data flow from scraper to API"""
        
        match_id = 'test_match_odds'
        
        # Step 1: Store current odds
        success = schema_manager.set_with_ttl(
            KeyType.ODDS, 'aggregated',
            sample_odds_data,
            match_id=match_id
        )
        assert success
        
        # Step 2: Store odds history
        odds_history = [
            {
                'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
                'average': {'team1': 1.80, 'team2': 2.00}
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat(),
                'average': {'team1': 1.83, 'team2': 1.97}
            }
        ]
        
        success = schema_manager.set_with_ttl(
            KeyType.ODDS, 'history',
            odds_history,
            match_id=match_id
        )
        assert success
        
        # Step 3: Test API endpoints
        
        # Get current odds
        response = api_client.get(f"/odds/{match_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data['match_id'] == match_id
        assert 'current_odds' in data
        assert data['current_odds']['average']['team1'] == 1.85
        assert data['current_odds']['num_sources'] == 3
        
        # Get odds with history
        response = api_client.get(f"/odds/{match_id}?include_history=true")
        assert response.status_code == 200
        
        data = response.json()
        assert 'odds_history' in data
        assert len(data['odds_history']) == 2
    
    @pytest.mark.asyncio
    async def test_health_monitoring_flow(self, schema_manager, api_client):
        """Test health monitoring flow"""
        
        # Step 1: Store service health data
        services = ['prediction_tracker', 'scraper', 'odds_fetcher']
        
        for service in services:
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'uptime_seconds': 3600,
                    'requests_processed': 150,
                    'error_rate': 0.02
                }
            }
            
            success = schema_manager.set_with_ttl(
                KeyType.HEALTH, 'service',
                health_data,
                service_name=service
            )
            assert success
        
        # Step 2: Store system health
        system_health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_services': len(services),
            'total_services': len(services)
        }
        
        success = schema_manager.set_with_ttl(
            KeyType.HEALTH, 'system',
            system_health
        )
        assert success
        
        # Step 3: Test health endpoints
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] in ['healthy', 'degraded']
        assert 'services' in data
        assert 'redis_health' in data
        
        # Test system status
        response = api_client.get("/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'services' in data
        assert data['services']['total'] == len(services)
    
    @pytest.mark.asyncio
    async def test_metrics_flow(self, schema_manager, api_client, sample_performance_metrics):
        """Test metrics data flow"""
        
        # Step 1: Store performance metrics
        success = schema_manager.set_with_ttl(
            KeyType.METRICS, 'performance',
            sample_performance_metrics
        )
        assert success
        
        # Step 2: Store daily metrics for different services
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        scraper_metrics = {
            'total_requests': 1000,
            'successful_requests': 950,
            'failed_requests': 50,
            'avg_response_time': 1.2,
            'matches_scraped': 45
        }
        
        success = schema_manager.set_with_ttl(
            KeyType.METRICS, 'scraper',
            scraper_metrics,
            date=date_str
        )
        assert success
        
        # Step 3: Test metrics endpoints
        
        # Get performance metrics
        response = api_client.get("/metrics/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert 'metrics' in data
        assert data['metrics']['total_bets'] == 50
        assert data['metrics']['hit_rate'] == 64.0
        
        # Get service-specific metrics
        response = api_client.get(f"/metrics/scraper?date={date_str}")
        assert response.status_code == 200
        
        data = response.json()
        assert data['service'] == 'scraper'
        assert data['date'] == date_str
        assert 'metrics' in data
        assert data['metrics']['total_requests'] == 1000
    
    @pytest.mark.asyncio
    async def test_redis_operations_flow(self, schema_manager, api_client):
        """Test Redis operations and management"""
        
        # Step 1: Populate Redis with various data types
        test_data = {
            'predictions': 5,
            'signals': 3,
            'matches': 8,
            'odds': 12
        }
        
        for data_type, count in test_data.items():
            for i in range(count):
                key = f"{data_type}:test_{i}"
                value = {'id': i, 'type': data_type, 'timestamp': datetime.now().isoformat()}
                schema_manager.redis.set(key, json.dumps(value))
        
        # Step 2: Test Redis stats endpoint
        response = api_client.get("/redis/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert 'memory_usage' in data
        assert 'redis_info' in data
        assert 'queue_stats' in data
        
        redis_info = data['redis_info']
        assert 'version' in redis_info
        assert 'connected_clients' in redis_info
        
        # Step 3: Test Redis cleanup
        response = api_client.post("/redis/cleanup")
        assert response.status_code == 200
        
        data = response.json()
        assert data['message'] == 'Cleanup completed'
        assert 'cleanup_stats' in data
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, schema_manager, api_client):
        """Test search functionality across the system"""
        
        # Step 1: Store searchable predictions
        predictions = [
            {
                'match_id': 'search_test_1',
                'teams': [{'name': 'Natus Vincere'}, {'name': 'FaZe Clan'}],
                'prediction': {'confidence': 0.75, 'expected_value': 0.08}
            },
            {
                'match_id': 'search_test_2',
                'teams': [{'name': 'G2 Esports'}, {'name': 'Astralis'}],
                'prediction': {'confidence': 0.65, 'expected_value': 0.05}
            },
            {
                'match_id': 'search_test_3',
                'teams': [{'name': 'Natus Vincere'}, {'name': 'G2 Esports'}],
                'prediction': {'confidence': 0.82, 'expected_value': 0.12}
            }
        ]
        
        for pred in predictions:
            schema_manager.redis.lpush(
                'predictions:recent',
                json.dumps(pred, default=str)
            )
        
        # Step 2: Test search by team
        search_query = {
            'team': 'Natus Vincere',
            'limit': 10
        }
        
        response = api_client.post("/predictions/search", json=search_query)
        assert response.status_code == 200
        
        data = response.json()
        assert data['count'] >= 2  # Should find 2 matches with Natus Vincere
        
        # Verify results contain the team
        for prediction in data['predictions']:
            team_names = [team['name'] for team in prediction['teams']]
            assert 'Natus Vincere' in team_names
        
        # Step 3: Test search by confidence
        search_query = {
            'min_confidence': 0.7,
            'limit': 10
        }
        
        response = api_client.post("/predictions/search", json=search_query)
        assert response.status_code == 200
        
        data = response.json()
        # Should find predictions with confidence >= 0.7
        for prediction in data['predictions']:
            assert prediction['prediction']['confidence'] >= 0.7
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, schema_manager, api_client):
        """Test error handling and system resilience"""
        
        # Test 1: Non-existent prediction
        response = api_client.get("/predictions/non_existent_match")
        assert response.status_code == 404
        assert "not found" in response.json()['detail'].lower()
        
        # Test 2: Non-existent odds
        response = api_client.get("/odds/non_existent_match")
        assert response.status_code == 404
        assert "not found" in response.json()['detail'].lower()
        
        # Test 3: Invalid service name for metrics
        response = api_client.get("/metrics/invalid_service")
        assert response.status_code == 400
        assert "Invalid service" in response.json()['detail']
        
        # Test 4: Search with no results
        search_query = {
            'team': 'NonExistentTeam',
            'limit': 10
        }
        
        response = api_client.post("/predictions/search", json=search_query)
        assert response.status_code == 200
        
        data = response.json()
        assert data['count'] == 0
        assert len(data['predictions']) == 0
        
        # Test 5: Empty queues
        response = api_client.get("/signals/queue")
        assert response.status_code == 200
        
        data = response.json()
        # Should handle empty queues gracefully
        assert 'queue_stats' in data
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, schema_manager, api_client):
        """Test concurrent operations on the system"""
        
        async def store_prediction(i):
            """Store a prediction concurrently"""
            prediction = {
                'match_id': f'concurrent_test_{i}',
                'teams': [{'name': f'Team_{i}_A'}, {'name': f'Team_{i}_B'}],
                'prediction': {'confidence': 0.6 + (i % 4) * 0.1}
            }
            
            success = schema_manager.set_with_ttl(
                KeyType.PREDICTION, 'current',
                prediction,
                match_id=prediction['match_id']
            )
            return success
        
        async def fetch_predictions():
            """Fetch predictions concurrently"""
            response = api_client.get("/predictions/recent?limit=20")
            return response.status_code == 200
        
        # Run concurrent operations
        store_tasks = [store_prediction(i) for i in range(10)]
        fetch_tasks = [fetch_predictions() for _ in range(5)]
        
        store_results = await asyncio.gather(*store_tasks)
        fetch_results = await asyncio.gather(*fetch_tasks)
        
        # Verify all operations succeeded
        assert all(store_results)
        assert all(fetch_results)
        
        # Verify data integrity
        response = api_client.get("/predictions/recent?limit=50")
        assert response.status_code == 200
        
        data = response.json()
        concurrent_predictions = [
            p for p in data['predictions'] 
            if p['match_id'].startswith('concurrent_test_')
        ]
        
        # Should have stored all concurrent predictions
        assert len(concurrent_predictions) >= 10
