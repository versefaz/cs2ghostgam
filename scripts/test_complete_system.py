#!/usr/bin/env python3
"""
Complete System Test - End-to-End CS2 Betting System Validation
Tests all components: Pipeline, Reporter, Scrapers, Models, and Publishers
"""

import os
import sys
import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_test.log')
    ]
)

logger = logging.getLogger(__name__)


class SystemTester:
    """Comprehensive system testing suite"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.utcnow()
    
    async def run_all_tests(self):
        """Run complete system test suite"""
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE CS2 BETTING SYSTEM TEST")
        logger.info("=" * 60)
        
        tests = [
            ("Core Components Test", self.test_core_components),
            ("Reporter System Test", self.test_reporter_system),
            ("Pipeline Integration Test", self.test_pipeline_integration),
            ("Mock Match Simulation", self.test_mock_match_simulation),
            ("Performance Metrics Test", self.test_performance_metrics),
            ("System Cleanup Test", self.test_system_cleanup)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n🧪 Running: {test_name}")
                result = await test_func()
                self.test_results[test_name] = {"status": "PASS" if result else "FAIL", "details": result}
                logger.info(f"✅ {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                logger.error(f"❌ {test_name}: FAIL - {e}")
                self.test_results[test_name] = {"status": "FAIL", "details": str(e)}
        
        await self.generate_test_report()
    
    async def test_core_components(self):
        """Test core system components"""
        try:
            # Test imports
            from core.integrated_pipeline import IntegratedPipeline
            from core.match_reporter import get_reporter
            from core.pubsub.publisher import get_publisher
            from core.session.session_manager import session_manager
            
            logger.info("✓ All core imports successful")
            
            # Test reporter initialization
            reporter = get_reporter()
            assert reporter is not None, "Reporter initialization failed"
            logger.info("✓ Reporter initialized successfully")
            
            # Test publisher initialization (with fallback)
            publisher = await get_publisher(
                url="redis://localhost:6379",
                enabled=False,  # Use NullPublisher for testing
                connect_timeout=1.0
            )
            assert publisher is not None, "Publisher initialization failed"
            logger.info(f"✓ Publisher initialized: {publisher.mode}")
            
            return True
            
        except Exception as e:
            logger.error(f"Core components test failed: {e}")
            return False
    
    async def test_reporter_system(self):
        """Test match reporter functionality"""
        try:
            from core.match_reporter import get_reporter
            
            reporter = get_reporter()
            
            # Test match report creation
            match_data = {
                'match_id': 'test_match_001',
                'team1': 'Natus Vincere',
                'team2': 'FaZe Clan',
                'time': '2024-01-15 18:00:00',
                'event': 'Test Tournament'
            }
            
            prediction_data = {
                'predicted_winner': 'Natus Vincere',
                'confidence': 0.75,
                'map_predictions': {'dust2': 0.8, 'mirage': 0.7}
            }
            
            odds_data = {
                'team1_odds': 1.85,
                'team2_odds': 2.15,
                'market_analysis': {'volume': 'high', 'trend': 'stable'}
            }
            
            # Create match report
            match_id = await reporter.create_match_report(match_data, prediction_data, odds_data)
            assert match_id == 'test_match_001', "Match report creation failed"
            logger.info(f"✓ Match report created: {match_id}")
            
            # Test match status update
            await reporter.update_match_status(match_id, 'completed', 'Natus Vincere')
            logger.info("✓ Match status updated successfully")
            
            # Test performance summary
            summary = await reporter.get_performance_summary()
            assert isinstance(summary, dict), "Performance summary failed"
            logger.info(f"✓ Performance summary: {summary.get('accuracy_rate', 'N/A')}")
            
            # Test daily report generation
            daily_report = await reporter.generate_daily_report()
            assert isinstance(daily_report, dict), "Daily report generation failed"
            logger.info("✓ Daily report generated successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Reporter system test failed: {e}")
            return False
    
    async def test_pipeline_integration(self):
        """Test integrated pipeline functionality"""
        try:
            from core.integrated_pipeline import IntegratedPipeline
            
            # Create pipeline instance
            pipeline = IntegratedPipeline(cfg={})
            
            # Test pipeline initialization
            await pipeline.start()
            logger.info("✓ Pipeline started successfully")
            
            # Wait a bit to let background tasks run
            await asyncio.sleep(5)
            
            # Check that all components are initialized
            assert pipeline.publisher is not None, "Publisher not initialized"
            assert pipeline.fe is not None, "Feature engineer not initialized"
            assert pipeline.reporter is not None, "Reporter not initialized"
            assert pipeline._started is True, "Pipeline not marked as started"
            
            logger.info("✓ All pipeline components initialized")
            logger.info(f"✓ Background tasks running: {len(pipeline._bg_tasks)}")
            
            # Test graceful shutdown
            await pipeline.stop()
            logger.info("✓ Pipeline stopped gracefully")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline integration test failed: {e}")
            return False
    
    async def test_mock_match_simulation(self):
        """Simulate a complete match lifecycle"""
        try:
            from core.match_reporter import get_reporter
            
            reporter = get_reporter()
            
            # Simulate multiple matches
            matches = [
                {
                    'match_id': 'sim_001',
                    'team1': 'Astralis',
                    'team2': 'G2 Esports',
                    'predicted_winner': 'Astralis',
                    'actual_winner': 'Astralis',
                    'confidence': 0.82
                },
                {
                    'match_id': 'sim_002', 
                    'team1': 'ENCE',
                    'team2': 'Vitality',
                    'predicted_winner': 'Vitality',
                    'actual_winner': 'ENCE',
                    'confidence': 0.68
                },
                {
                    'match_id': 'sim_003',
                    'team1': 'Cloud9',
                    'team2': 'Liquid',
                    'predicted_winner': 'Liquid',
                    'actual_winner': 'Liquid',
                    'confidence': 0.91
                }
            ]
            
            # Process each match
            for match in matches:
                # Create match report
                match_data = {
                    'match_id': match['match_id'],
                    'team1': match['team1'],
                    'team2': match['team2'],
                    'event': 'Simulation Tournament'
                }
                
                prediction_data = {
                    'predicted_winner': match['predicted_winner'],
                    'confidence': match['confidence']
                }
                
                odds_data = {
                    'team1_odds': 1.9,
                    'team2_odds': 2.1
                }
                
                # Create and complete match
                match_id = await reporter.create_match_report(match_data, prediction_data, odds_data)
                await reporter.update_match_status(match_id, 'completed', match['actual_winner'])
                
                logger.info(f"✓ Simulated match: {match['team1']} vs {match['team2']} - Winner: {match['actual_winner']}")
            
            # Check final performance
            summary = await reporter.get_performance_summary()
            logger.info(f"✓ Simulation complete - Final accuracy: {summary.get('accuracy_rate', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Mock match simulation failed: {e}")
            return False
    
    async def test_performance_metrics(self):
        """Test performance metrics calculation"""
        try:
            from core.match_reporter import get_reporter
            
            reporter = get_reporter()
            
            # Get current metrics
            summary = await reporter.get_performance_summary()
            
            # Validate metrics structure
            required_fields = ['total_predictions', 'accuracy_rate', 'roi', 'profit_loss', 'current_streak']
            for field in required_fields:
                assert field in summary, f"Missing metric field: {field}"
            
            logger.info("✓ All required metric fields present")
            
            # Test daily report
            daily_report = await reporter.generate_daily_report()
            assert 'daily_performance' in daily_report, "Daily performance missing"
            assert 'overall_performance' in daily_report, "Overall performance missing"
            
            logger.info("✓ Daily report structure validated")
            
            # Test cleanup functionality
            await reporter.cleanup_old_reports(days_to_keep=1)
            logger.info("✓ Report cleanup completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance metrics test failed: {e}")
            return False
    
    async def test_system_cleanup(self):
        """Test system cleanup and resource management"""
        try:
            from core.session.session_manager import session_manager
            
            # Test session cleanup
            await session_manager.close_all()
            logger.info("✓ Session manager cleanup completed")
            
            # Test file system cleanup
            reports_dir = Path("reports")
            if reports_dir.exists():
                logger.info(f"✓ Reports directory exists with {len(list(reports_dir.glob('*')))} files")
            
            return True
            
        except Exception as e:
            logger.error(f"System cleanup test failed: {e}")
            return False
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.utcnow()
        duration = end_time - self.start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("SYSTEM TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"Duration: {duration.total_seconds():.2f} seconds")
        
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            logger.info(f"{status_icon} {test_name}: {result['status']}")
        
        # Save test report
        report_data = {
            'test_run_id': f"test_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'test_results': self.test_results
        }
        
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"system_test_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📊 Test report saved: {report_file}")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED! System is ready for production.")
        else:
            logger.warning(f"\n⚠️  {total_tests - passed_tests} tests failed. Review issues before deployment.")


async def main():
    """Main test execution"""
    try:
        # Set environment variables for testing
        os.environ['REDIS_ENABLED'] = 'false'  # Use NullPublisher for testing
        os.environ['FEATURE_CONFIG'] = 'configs/features/cs2.yaml'
        
        tester = SystemTester()
        await tester.run_all_tests()
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
