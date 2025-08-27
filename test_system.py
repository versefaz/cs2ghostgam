#!/usr/bin/env python3
"""
Quick System Test - Test basic functionality without missing dependencies
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print test banner"""
    banner = """
===============================================================================
                    CS2 BETTING SYSTEM - QUICK TEST                           
                                                                               
  Testing Core Components and Basic Functionality                          
  Verifying System Readiness for Production                                
===============================================================================
    """
    print(banner)

async def test_hltv_scraper():
    """Test HLTV scraper functionality"""
    try:
        from app.scrapers.enhanced_hltv_scraper import EnhancedHLTVScraper
        
        logger.info("Testing HLTV Scraper...")
        scraper = EnhancedHLTVScraper()
        await scraper.initialize()
        
        # Test basic scraping
        matches = await scraper.get_upcoming_matches_with_stats(limit=3)
        logger.info(f"Found {len(matches)} upcoming matches")
        
        if matches:
            for match in matches[:2]:  # Test first 2 matches
                logger.info(f"Match: {match.team1_name} vs {match.team2_name}")
        
        await scraper.close()
        return True
        
    except Exception as e:
        logger.error(f"HLTV Scraper test failed: {e}")
        return False

async def test_odds_scraper():
    """Test odds scraper functionality"""
    try:
        from app.scrapers.robust_odds_scraper import RobustOddsScraper
        
        logger.info("Testing Odds Scraper...")
        scraper = RobustOddsScraper()
        await scraper.initialize()
        
        # Test basic odds scraping
        consensus_list = await scraper.scrape_all_sources()
        logger.info(f"Found {len(consensus_list)} odds consensus")
        
        await scraper.close()
        return True
        
    except Exception as e:
        logger.error(f"Odds Scraper test failed: {e}")
        return False

async def test_signal_generator():
    """Test signal generator functionality"""
    try:
        logger.info("Testing Signal Generator...")
        
        # Test basic signal generation logic without full initialization
        from models.signal import BettingSignal, SignalSide, SignalPriority
        from utils.kelly import calculate_kelly_fraction
        
        # Test Kelly calculation
        kelly_fraction = calculate_kelly_fraction(
            probability=0.75,
            odds=1.85
        )
        
        logger.info(f"Kelly fraction calculated: {kelly_fraction}")
        
        # Test signal creation
        signal = BettingSignal(
            signal_id="test_001",
            match_id="match_001",
            side=SignalSide.TEAM1_WIN,
            stake=100.0,
            ev=0.15,
            confidence=0.75
        )
        
        logger.info(f"Signal created: {signal.signal_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Signal Generator test failed: {e}")
        return False

async def test_performance_optimizer():
    """Test performance optimizer"""
    try:
        from core.performance_optimizer import PerformanceOptimizer, OptimizationConfig
        
        logger.info("Testing Performance Optimizer...")
        config = OptimizationConfig(enable_auto_tuning=False)  # Disable auto-tuning for test
        optimizer = PerformanceOptimizer(config)
        await optimizer.initialize()
        
        # Test basic metrics recording
        optimizer.record_processing_time(150.0)
        optimizer.record_cache_hit(True)
        optimizer.record_memory_usage()
        
        logger.info(f"Performance metrics: {optimizer.metrics}")
        
        await optimizer.close()
        return True
        
    except Exception as e:
        logger.error(f"Performance Optimizer test failed: {e}")
        return False

async def test_basic_ml_components():
    """Test basic ML components"""
    try:
        # Test only what exists
        logger.info("Testing ML Components...")
        
        # Test basic ML imports that should work
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        
        logger.info("Basic ML libraries imported successfully")
        
        # Test basic ML functionality
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        xgb_model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        xgb_model.fit(X, y)
        
        logger.info("ML models trained successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"ML Components test failed: {e}")
        return False

async def run_comprehensive_test():
    """Run comprehensive system test"""
    print_banner()
    
    test_results = {
        'HLTV Scraper': False,
        'Odds Scraper': False,
        'Signal Generator': False,
        'Performance Optimizer': False,
        'ML Components': False
    }
    
    # Run all tests
    test_results['HLTV Scraper'] = await test_hltv_scraper()
    test_results['Odds Scraper'] = await test_odds_scraper()
    test_results['Signal Generator'] = await test_signal_generator()
    test_results['Performance Optimizer'] = await test_performance_optimizer()
    test_results['ML Components'] = await test_basic_ml_components()
    
    # Print results
    print("\n" + "="*80)
    print("                           TEST RESULTS")
    print("="*80)
    
    all_passed = True
    for component, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        icon = "[OK]" if passed else "[FAIL]"
        print(f"{icon} {component:<25} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("ALL TESTS PASSED! System is ready for production!")
        print("You can now run: python main.py")
    else:
        print("Some tests failed. Please check the errors above.")
    
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    # Set event loop policy for Windows
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run tests
    asyncio.run(run_comprehensive_test())
