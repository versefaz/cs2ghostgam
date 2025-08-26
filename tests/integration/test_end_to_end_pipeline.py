#!/usr/bin/env python3
"""
End-to-End Integration Tests for CS2 Betting Pipeline
Tests complete data flow: Match Data → Features → Prediction → Signal → Redis → Consumer
"""

import asyncio
import pytest
import json
import redis.asyncio as redis
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Core components
from core.integrated_pipeline import IntegratedPipeline, PipelineConfig
from app.scrapers.enhanced_hltv_scraper import MatchContext, TeamStats
from app.scrapers.robust_odds_scraper import MarketConsensus
from core.production_signal_generator import Signal, SignalType
from shared.redis_schema import RedisSchema


class TestEndToEndPipeline:
    """Comprehensive end-to-end pipeline integration tests"""
    
    @pytest.fixture
    async def redis_client(self):
        """Redis client for testing"""
        client = redis.Redis.from_url("redis://localhost:6379/15")  # Test DB
        await client.flushdb()
        yield client
        await client.flushdb()
        await client.close()
    
    @pytest.fixture
    def pipeline_config(self):
        """Test pipeline configuration"""
        return PipelineConfig(
            match_scrape_interval=10,  # Faster for testing
            odds_scrape_interval=5,
            max_concurrent_matches=3,
            min_confidence_threshold=0.5,  # Lower for testing
            redis_url="redis://localhost:6379/15"
        )
    
    @pytest.fixture
    def mock_match_context(self):
        """Mock match context data"""
        team1_stats = TeamStats(
            team_name="Team Liquid",
            world_ranking=3,
            ranking_points=850,
            recent_form=[1, 1, 0, 1, 1],  # W-W-L-W-W
            win_rate_30d=0.75,
            avg_rating=1.15,
            kd_ratio=1.08,
            map_stats={
                "dust2": {"wins": 8, "losses": 2, "winrate": 0.8},
                "mirage": {"wins": 7, "losses": 3, "winrate": 0.7}
            },
            h2h_wins=3,
            country="USA",
            logo_url="https://example.com/liquid.png"
        )
        
        team2_stats = TeamStats(
            team_name="Astralis",
            world_ranking=7,
            ranking_points=720,
            recent_form=[1, 0, 1, 0, 1],  # W-L-W-L-W
            win_rate_30d=0.60,
            avg_rating=1.05,
            kd_ratio=1.02,
            map_stats={
                "dust2": {"wins": 6, "losses": 4, "winrate": 0.6},
                "mirage": {"wins": 5, "losses": 5, "winrate": 0.5}
            },
            h2h_wins=2,
            country="Denmark",
            logo_url="https://example.com/astralis.png"
        )
        
        return MatchContext(
            match_id="test_match_001",
            team1_name="Team Liquid",
            team2_name="Astralis",
            team1_stats=team1_stats,
            team2_stats=team2_stats,
            event_name="ESL Pro League",
            match_time=datetime.utcnow() + timedelta(hours=2),
            bo_format="BO3",
            map_pool=["dust2", "mirage", "inferno"]
        )
    
    @pytest.fixture
    def mock_odds_data(self):
        """Mock odds data"""
        return {
            'match_winner': {
                'team liquid': 1.85,
                'astralis': 1.95
            },
            'best_odds': {
                'team liquid': 1.90,
                'astralis': 2.00
            },
            'bookmaker_count': 5,
            'arbitrage_opportunity': False,
            'arbitrage_profit': 0.0,
            'sources': ['oddsportal', 'ggbet'],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, pipeline_config, redis_client, mock_match_context, mock_odds_data):
        """Test complete end-to-end pipeline flow"""
        pipeline = IntegratedPipeline(pipeline_config)
        
        try:
            # Mock all external dependencies
            with patch.multiple(
                pipeline,
                hltv_scraper=AsyncMock(),
                odds_scraper=AsyncMock(),
                signal_generator=AsyncMock(),
                prediction_model=AsyncMock(),
                feature_engineer=MagicMock(),
                redis_publisher=AsyncMock(),
                redis_schema=AsyncMock()
            ):
                # Setup mocks
                pipeline.hltv_scraper.get_upcoming_matches_with_stats.return_value = [mock_match_context]
                pipeline.odds_scraper.get_odds_for_match.return_value = MagicMock(
                    consensus_odds=mock_odds_data['match_winner'],
                    best_odds=mock_odds_data['best_odds'],
                    bookmaker_count=mock_odds_data['bookmaker_count'],
                    arbitrage_opportunity=mock_odds_data['arbitrage_opportunity'],
                    arbitrage_profit=mock_odds_data['arbitrage_profit'],
                    sources=mock_odds_data['sources'],
                    timestamp=mock_odds_data['timestamp']
                )
                
                # Mock feature extraction
                pipeline.feature_engineer.extract_features.return_value = [0.75, 0.60, 1.85, 1.95, 0.8, 0.6]
                
                # Mock prediction
                pipeline.prediction_model.predict_match_outcome.return_value = {
                    'predicted_winner': 'Team Liquid',
                    'confidence': 0.72,
                    'probabilities': {'Team Liquid': 0.72, 'Astralis': 0.28},
                    'model_version': '1.0'
                }
                
                # Mock signal generation
                mock_signal = Signal(
                    signal_id="test_signal_001",
                    match_id="test_match_001",
                    signal_type=SignalType.VALUE_BET,
                    team_name="Team Liquid",
                    recommended_bet="Team Liquid to win",
                    confidence=0.72,
                    stake_amount=50.0,
                    expected_odds=1.85,
                    expected_return=92.5,
                    reasoning="Strong team form and ranking advantage",
                    timestamp=datetime.utcnow()
                )
                pipeline.signal_generator.generate_signals.return_value = [mock_signal]
                
                # Initialize pipeline components
                await pipeline.initialize()
                
                # Process one complete cycle
                await pipeline._scrape_and_process_matches()
                
                # Verify the complete flow
                assert pipeline.metrics.matches_processed >= 1
                assert pipeline.metrics.predictions_made >= 1
                assert pipeline.metrics.signals_generated >= 1
                
                # Verify all components were called
                pipeline.hltv_scraper.get_upcoming_matches_with_stats.assert_called()
                pipeline.odds_scraper.get_odds_for_match.assert_called()
                pipeline.feature_engineer.extract_features.assert_called()
                pipeline.prediction_model.predict_match_outcome.assert_called()
                pipeline.signal_generator.generate_signals.assert_called()
                pipeline.redis_publisher.publish_signal.assert_called()
                
        finally:
            await pipeline.close()
    
    @pytest.mark.asyncio
    async def test_redis_integration(self, pipeline_config, redis_client):
        """Test Redis integration and data flow"""
        redis_schema = RedisSchema(redis_url=pipeline_config.redis_url)
        await redis_schema.initialize()
        
        try:
            # Test signal storage
            signal_data = {
                'signal_id': 'test_signal_redis',
                'match_id': 'test_match_redis',
                'signal_type': 'VALUE_BET',
                'team_name': 'Team Liquid',
                'confidence': 0.75,
                'stake_amount': 100.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await redis_schema.store_signal(signal_data)
            
            # Verify signal was stored
            stored_signals = await redis_schema.get_recent_signals(limit=1)
            assert len(stored_signals) == 1
            assert stored_signals[0]['signal_id'] == 'test_signal_redis'
            
            # Test match data storage
            match_data = {
                'match_id': 'test_match_redis',
                'team1_name': 'Team Liquid',
                'team2_name': 'Astralis',
                'event_name': 'Test Event',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await redis_schema.store_match(match_data)
            
            # Verify match was stored
            stored_match = await redis_schema.get_match('test_match_redis')
            assert stored_match['team1_name'] == 'Team Liquid'
            
        finally:
            await redis_schema.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, pipeline_config):
        """Test error handling and recovery mechanisms"""
        pipeline = IntegratedPipeline(pipeline_config)
        
        try:
            with patch.multiple(
                pipeline,
                hltv_scraper=AsyncMock(),
                odds_scraper=AsyncMock(),
                signal_generator=AsyncMock(),
                redis_publisher=AsyncMock()
            ):
                # Simulate scraper failure
                pipeline.hltv_scraper.get_upcoming_matches_with_stats.side_effect = Exception("Scraper failed")
                
                # Initialize pipeline
                await pipeline.initialize()
                
                # Process should handle error gracefully
                await pipeline._scrape_and_process_matches()
                
                # Verify error was recorded
                assert pipeline.metrics.errors_encountered > 0
                
                # Verify pipeline continues running
                assert pipeline.is_running
                
        finally:
            await pipeline.close()
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, pipeline_config, mock_match_context):
        """Test performance metrics collection"""
        pipeline = IntegratedPipeline(pipeline_config)
        
        try:
            with patch.multiple(
                pipeline,
                hltv_scraper=AsyncMock(),
                odds_scraper=AsyncMock(),
                signal_generator=AsyncMock(),
                redis_publisher=AsyncMock(),
                redis_schema=AsyncMock()
            ):
                # Setup successful processing
                pipeline.hltv_scraper.get_upcoming_matches_with_stats.return_value = [mock_match_context]
                pipeline.odds_scraper.get_odds_for_match.return_value = MagicMock(
                    consensus_odds={'team liquid': 1.85, 'astralis': 1.95},
                    best_odds={'team liquid': 1.90, 'astralis': 2.00},
                    bookmaker_count=3,
                    arbitrage_opportunity=False,
                    arbitrage_profit=0.0,
                    sources=['test'],
                    timestamp=datetime.utcnow().isoformat()
                )
                
                await pipeline.initialize()
                
                # Process multiple cycles
                for _ in range(3):
                    await pipeline._scrape_and_process_matches()
                
                # Verify metrics
                assert pipeline.metrics.matches_processed >= 3
                assert pipeline.metrics.average_processing_time > 0
                assert pipeline.metrics.last_successful_run is not None
                
        finally:
            await pipeline.close()
    
    @pytest.mark.asyncio
    async def test_signal_consumer_integration(self, pipeline_config, redis_client):
        """Test signal consumer integration"""
        # This would test the Discord/Telegram bot consumers
        # For now, we'll test Redis pub/sub functionality
        
        redis_schema = RedisSchema(redis_url=pipeline_config.redis_url)
        await redis_schema.initialize()
        
        try:
            # Setup subscriber
            pubsub = redis_client.pubsub()
            await pubsub.subscribe('cs2_signals')
            
            # Publish a test signal
            test_signal = {
                'signal_id': 'test_consumer_signal',
                'match_id': 'test_match_consumer',
                'signal_type': 'VALUE_BET',
                'team_name': 'Team Liquid',
                'confidence': 0.80,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await redis_client.publish('cs2_signals', json.dumps(test_signal))
            
            # Verify message was received
            message = await pubsub.get_message(timeout=5.0)
            if message and message['type'] == 'message':
                received_signal = json.loads(message['data'])
                assert received_signal['signal_id'] == 'test_consumer_signal'
            
            await pubsub.unsubscribe('cs2_signals')
            
        finally:
            await redis_schema.close()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, pipeline_config):
        """Test health monitoring and alerting"""
        pipeline = IntegratedPipeline(pipeline_config)
        
        try:
            with patch.multiple(
                pipeline,
                alert_system=AsyncMock(),
                redis_schema=AsyncMock()
            ):
                await pipeline.initialize()
                
                # Simulate high error rate
                pipeline.metrics.errors_encountered = 10
                pipeline.metrics.matches_processed = 50
                
                # Run health check
                await pipeline._perform_health_checks()
                
                # Verify health status was stored
                pipeline.redis_schema.store_health_status.assert_called()
                
                # Verify alert was sent for high error rate
                pipeline.alert_system.send_alert.assert_called()
                
        finally:
            await pipeline.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, pipeline_config):
        """Test concurrent match processing"""
        pipeline = IntegratedPipeline(pipeline_config)
        
        # Create multiple mock matches
        mock_matches = []
        for i in range(5):
            match = MagicMock()
            match.match_id = f"test_match_{i}"
            match.team1_name = f"Team{i}A"
            match.team2_name = f"Team{i}B"
            mock_matches.append(match)
        
        try:
            with patch.multiple(
                pipeline,
                hltv_scraper=AsyncMock(),
                _process_match_end_to_end=AsyncMock()
            ):
                pipeline.hltv_scraper.get_upcoming_matches_with_stats.return_value = mock_matches
                pipeline._process_match_end_to_end.return_value = [MagicMock()]
                
                await pipeline.initialize()
                
                # Process all matches
                await pipeline._scrape_and_process_matches()
                
                # Verify all matches were processed
                assert pipeline._process_match_end_to_end.call_count == len(mock_matches)
                
        finally:
            await pipeline.close()


class TestPipelineStressTest:
    """Stress tests for pipeline performance"""
    
    @pytest.mark.asyncio
    async def test_high_volume_processing(self, pipeline_config):
        """Test pipeline under high volume"""
        pipeline = IntegratedPipeline(pipeline_config)
        
        # Create many mock matches
        mock_matches = [MagicMock() for _ in range(50)]
        for i, match in enumerate(mock_matches):
            match.match_id = f"stress_test_match_{i}"
        
        try:
            with patch.multiple(
                pipeline,
                hltv_scraper=AsyncMock(),
                _process_match_end_to_end=AsyncMock()
            ):
                pipeline.hltv_scraper.get_upcoming_matches_with_stats.return_value = mock_matches
                pipeline._process_match_end_to_end.return_value = [MagicMock()]
                
                await pipeline.initialize()
                
                start_time = datetime.utcnow()
                await pipeline._scrape_and_process_matches()
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Verify performance
                assert processing_time < 60  # Should complete within 60 seconds
                assert pipeline.metrics.matches_processed >= len(mock_matches)
                
        finally:
            await pipeline.close()
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, pipeline_config):
        """Test memory usage under load"""
        import psutil
        import os
        
        pipeline = IntegratedPipeline(pipeline_config)
        process = psutil.Process(os.getpid())
        
        try:
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate processing many matches
            for _ in range(100):
                pipeline.processed_matches.add(f"memory_test_match_{_}")
                pipeline.odds_cache[f"odds_{_}"] = ({"test": "data"}, datetime.utcnow())
                pipeline.feature_cache[f"features_{_}"] = [1.0, 2.0, 3.0]
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB for test data)
            assert memory_increase < 100
            
        finally:
            await pipeline.close()


# Performance benchmarks
@pytest.mark.benchmark
class TestPipelineBenchmarks:
    """Benchmark tests for pipeline performance"""
    
    @pytest.mark.asyncio
    async def test_match_processing_benchmark(self, benchmark, pipeline_config, mock_match_context):
        """Benchmark match processing speed"""
        pipeline = IntegratedPipeline(pipeline_config)
        
        async def process_match():
            with patch.multiple(
                pipeline,
                _get_match_odds=AsyncMock(return_value={"test": "odds"}),
                _extract_match_features=AsyncMock(return_value=[1.0, 2.0]),
                _make_prediction=AsyncMock(return_value={"confidence": 0.8}),
                _generate_signals=AsyncMock(return_value=[MagicMock()]),
                _publish_signals=AsyncMock(return_value=[MagicMock()]),
                _store_match_data=AsyncMock()
            ):
                await pipeline._process_match_end_to_end(mock_match_context)
        
        try:
            await pipeline.initialize()
            
            # Benchmark the processing
            result = await benchmark(process_match)
            
            # Should process a match in under 1 second
            assert result is not None
            
        finally:
            await pipeline.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
