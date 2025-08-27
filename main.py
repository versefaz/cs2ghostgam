#!/usr/bin/env python3
"""
CS2 Betting System - Main Entry Point
Production-ready betting signal generation with comprehensive monitoring
"""

import asyncio
import logging
import signal
import sys
import os
import traceback
from datetime import datetime
from contextlib import suppress

# Core system imports
from core.integrated_pipeline import IntegratedPipeline
from core.performance_optimizer import PerformanceOptimizer
from core.session.session_manager import session_manager

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cs2_betting_system.log")
    ]
)

# Suppress noisy loggers for cleaner output
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global system state
pipeline = None
performance_optimizer = None
is_shutting_down = False

# Setup signal handlers
def signal_handler(signum, frame):
    global is_shutting_down
    logger.info(f"Received signal {signum}, initiating shutdown...")
    is_shutting_down = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Core integrated components
try:
    from monitoring.prometheus_metrics import PrometheusMetrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus metrics not available, using fallback")
try:
    from monitoring.alert_system import AlertSystem
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    print("Alert system not available, using fallback")
try:
    from shared.redis_schema import RedisSchema
    REDIS_SCHEMA_AVAILABLE = True
except ImportError:
    REDIS_SCHEMA_AVAILABLE = False
    print("Redis schema not available, using fallback")


logger = logging.getLogger(__name__)


def print_banner():
    """Print system banner"""
    banner = """
===============================================================================
                    CS2 BETTING SYSTEM V2.0 - PRODUCTION READY                
                                                                               
  Real-time Match Analysis    Advanced ML Predictions                   
  Automated Signal Generation Redis Pub/Sub Integration                 
  Performance Optimization    Comprehensive Monitoring                   
                                                                               
  Target: 80%+ Win Rate | 15%+ ROI | Real-time Processing                     
===============================================================================
"""
    print(banner)


class CS2BettingSystem:
    """Main CS2 Betting System orchestrator"""
    
    def __init__(self):
        self.pipeline: Optional[IntegratedPipeline] = None
        self.optimizer: Optional[PerformanceOptimizer] = None
        self.metrics = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Initialize metrics (with fallback)
        if PROMETHEUS_AVAILABLE:
            try:
                self.metrics = PrometheusMetrics(service_name="cs2_betting_system")
            except Exception as e:
                logger.warning(f"Prometheus metrics not available: {e}")
                self.metrics = None
        else:
            self.metrics = None
        
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
            
            # Initialize alert system (with fallback)
            if ALERT_SYSTEM_AVAILABLE:
                self.alert_system = AlertSystem()
                await self.alert_system.initialize()
            else:
                self.alert_system = None
            
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
    
    async def cleanup_system():
        """Enhanced cleanup system resources with proper session management"""
        global pipeline, performance_optimizer
        
        logger.info("🛑 Cleaning up system resources...")
        
        try:
            # Shutdown pipeline first
            if pipeline:
                logger.info("Shutting down integrated pipeline...")
                await pipeline.shutdown()
                pipeline = None
                
            # Close performance optimizer
            if performance_optimizer:
                logger.info("Closing performance optimizer...")
                await performance_optimizer.close()
                performance_optimizer = None
            
            # Close all HTTP sessions via SessionManager
            logger.info("Closing all HTTP sessions...")
            await session_manager.close_all()
            
            # Give time for all cleanup to complete
            await asyncio.sleep(0.5)
            
            # Cancel any remaining tasks
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if tasks:
                logger.info(f"Cancelling {len(tasks)} remaining tasks...")
                for task in tasks:
                    task.cancel()
                
                # Wait for tasks to cancel
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("✅ System cleanup completed")
    
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
    """Enhanced main application entry point with proper cleanup"""
    logging.getLogger(__name__).info("Starting CS2 Betting System v2.0...")
    pipeline = IntegratedPipeline(cfg={})
    await pipeline.start()
    
    # graceful shutdown
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop_event.set)
    
    await stop_event.wait()
    await pipeline.stop()


async def cleanup_system():
    """Enhanced cleanup system resources with proper session management"""
    global pipeline, performance_optimizer
    
    logger.info("Cleaning up system resources...")
    
    try:
        # Shutdown pipeline first
        if pipeline:
            logger.info("Shutting down integrated pipeline...")
            await pipeline.shutdown()
            pipeline = None
            
        # Close performance optimizer
        if performance_optimizer:
            logger.info("Closing performance optimizer...")
            await performance_optimizer.close()
            performance_optimizer = None
        
        # Close all HTTP sessions via SessionManager
        logger.info("Closing all HTTP sessions...")
        await session_manager.close_all()
        
        # Give time for all cleanup to complete
        await asyncio.sleep(0.5)
        
        # Cancel any remaining tasks
        tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} remaining tasks...")
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to cancel with timeout
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2.0
                )
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    logger.info("System cleanup completed")


if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Graceful exit on Ctrl+C