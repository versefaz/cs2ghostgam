#!/usr/bin/env python3
"""
CS2 Betting System - Production-Ready Main Entry Point
Complete Integration: Pipeline → Scrapers → ML → Signals → Redis → Consumers
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Core integrated components
from core.integrated_pipeline import IntegratedPipeline, PipelineConfig
from core.performance_optimizer import PerformanceOptimizer, OptimizationConfig
from monitoring.prometheus_metrics import PrometheusMetrics
from monitoring.alert_system import AlertSystem
from shared.redis_schema import RedisSchema


logger = logging.getLogger(__name__)


def print_banner():
    """Print system banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CS2 BETTING SYSTEM V2.0 - PRODUCTION READY                ║
║                                                                               ║
║  🎯 Complete Integration: Match Data → Features → ML → Signals → Redis        ║
║  🚀 Performance Optimized: Auto-tuning, Caching, Monitoring                  ║
║  🛡️  Production Grade: Error Handling, Health Checks, Alerting               ║
║  📊 Real-time Analytics: Prometheus, Grafana, Live Dashboard                 ║
║                                                                               ║
║  Target: 80%+ Win Rate | 15%+ ROI | 99.9% Uptime                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


class CS2BettingSystem:
    """Main CS2 Betting System orchestrator"""
    
    def __init__(self):
        self.pipeline: Optional[IntegratedPipeline] = None
        self.optimizer: Optional[PerformanceOptimizer] = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing CS2 Betting System...")
        
        try:
            # Initialize performance optimizer
            optimization_config = OptimizationConfig(
                max_cache_size=10000,
                max_concurrent_scrapers=5,
                enable_auto_tuning=True
            )
            self.optimizer = PerformanceOptimizer(optimization_config)
            await self.optimizer.initialize()
            
            # Initialize main pipeline
            pipeline_config = PipelineConfig(
                match_scrape_interval=300,
                odds_scrape_interval=180,
                max_concurrent_matches=10,
                min_confidence_threshold=0.65,
                redis_url="redis://localhost:6379",
                model_path="models/cs2_simple_model.pkl"
            )
            self.pipeline = IntegratedPipeline(pipeline_config)
            await self.pipeline.initialize()
            
            logger.info("CS2 Betting System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def run(self):
        """Run the complete CS2 betting system"""
        print_banner()
        logger.info("Starting CS2 Betting System...")
        
        try:
            # Initialize all components
            await self.initialize()
            self.is_running = True
            
            # Start all background tasks
            tasks = [
                asyncio.create_task(self.pipeline.run(), name="pipeline"),
                asyncio.create_task(self._monitoring_loop(), name="monitoring")
            ]
            
            logger.info("All system components started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"System runtime error: {e}")
        finally:
            await self.cleanup()
    
    async def shutdown(self):
        """Graceful system shutdown"""
        logger.info("Initiating graceful shutdown...")
        self.is_running = False
        self.shutdown_event.set()
    
    async def cleanup(self):
        """Cleanup all system resources"""
        logger.info("Cleaning up system resources...")
        
        try:
            # Close all components
            if self.pipeline:
                await self.pipeline.close()
            
            if self.optimizer:
                await self.optimizer.close()
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _monitoring_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                # Record performance metrics
                if self.optimizer:
                    self.optimizer.record_memory_usage()
                    self.optimizer.record_cpu_usage()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                await asyncio.sleep(60)


async def main():
    """Main entry point"""
    system = CS2BettingSystem()
    
    try:
        await system.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the system
    asyncio.run(main())
 