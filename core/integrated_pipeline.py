#!/usr/bin/env python3
"""
Integrated CS2 Betting Pipeline - Complete End-to-End System
Orchestrates: Match Data → Features → Prediction → Signal → Redis → Consumers
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

# Core components
from app.scrapers.enhanced_hltv_scraper import EnhancedHLTVScraper, MatchContext
from app.scrapers.robust_odds_scraper import RobustOddsScraper, MarketConsensus
from core.production_signal_generator import ProductionSignalGenerator
from ml_pipeline.training.model_trainer import ModelTrainer
from ml_pipeline.features.feature_engineering import FeatureEngineer
from cs2_betting_system.models.prediction_model import PredictionModel
from publishers.redis_publisher import RedisPublisher
from monitoring.prometheus_metrics import PrometheusMetrics
from monitoring.alert_system import AlertSystem
from shared.redis_schema import RedisSchema

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the integrated pipeline"""
    match_scrape_interval: int = 300  # 5 minutes
    odds_scrape_interval: int = 180   # 3 minutes
    max_concurrent_matches: int = 10
    max_signals_per_hour: int = 20
    min_confidence_threshold: float = 0.65
    redis_url: str = "redis://localhost:6379"
    model_path: str = "models/cs2_simple_model.pkl"
    retrain_interval_hours: int = 24


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    matches_processed: int = 0
    signals_generated: int = 0
    predictions_made: int = 0
    errors_encountered: int = 0
    average_processing_time: float = 0.0
    last_successful_run: Optional[datetime] = None
    uptime_start: datetime = None


class IntegratedPipeline:
    """Complete end-to-end CS2 betting pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.metrics = PipelineMetrics(uptime_start=datetime.utcnow())
        
        # Core components
        self.hltv_scraper: Optional[EnhancedHLTVScraper] = None
        self.odds_scraper: Optional[RobustOddsScraper] = None
        self.signal_generator: Optional[ProductionSignalGenerator] = None
        self.model_trainer: Optional[ModelTrainer] = None
        self.prediction_model: Optional[PredictionModel] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.redis_publisher: Optional[RedisPublisher] = None
        self.redis_schema: Optional[RedisSchema] = None
        
        # Monitoring
        self.prometheus_metrics = PrometheusMetrics()
        self.alert_system = AlertSystem()
        
        # State management
        self.is_running = False
        self.processed_matches = set()
        self.last_model_training = None
        
        # Caches
        self.team_stats_cache = {}
        self.odds_cache = {}
        self.feature_cache = {}
        
        # Background tasks
        self.background_tasks = []
    
    async def initialize(self):
        """Initialize all pipeline components"""
        logger.info("Initializing integrated CS2 betting pipeline...")
        
        try:
            # Initialize scrapers
            self.hltv_scraper = EnhancedHLTVScraper()
            await self.hltv_scraper.initialize()
            
            self.odds_scraper = RobustOddsScraper()
            await self.odds_scraper.initialize()
            
            # Initialize Redis components
            self.redis_schema = RedisSchema(redis_url=self.config.redis_url)
            await self.redis_schema.initialize()
            
            self.redis_publisher = RedisPublisher(redis_url=self.config.redis_url)
            await self.redis_publisher.initialize()
            
            # Initialize signal generator
            self.signal_generator = ProductionSignalGenerator(
                redis_url=self.config.redis_url,
                bankroll=10000.0
            )
            await self.signal_generator.initialize()
            
            # Initialize ML components
            self.feature_engineer = FeatureEngineer()
            self.model_trainer = ModelTrainer()
            
            # Load or train model
            if not self.model_trainer.load_model(self.config.model_path):
                logger.warning("Could not load existing model, will train new one")
                await self._train_initial_model()
            
            self.prediction_model = PredictionModel(
                model_path=self.config.model_path,
                feature_engineer=self.feature_engineer
            )
            
            # Initialize monitoring
            await self.alert_system.initialize()
            
            # Start background tasks
            self._start_background_tasks()
            
            logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            await self.alert_system.send_alert(
                "critical",
                "Pipeline Initialization Failed",
                f"Error: {str(e)}",
                {"component": "pipeline", "error": str(e)}
            )
            raise
    
    async def close(self):
        """Cleanup pipeline resources"""
        logger.info("Shutting down integrated pipeline...")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close components
        if self.hltv_scraper:
            await self.hltv_scraper.close()
        if self.odds_scraper:
            await self.odds_scraper.close()
        if self.signal_generator:
            await self.signal_generator.close()
        if self.redis_publisher:
            await self.redis_publisher.close()
        if self.redis_schema:
            await self.redis_schema.close()
        
        logger.info("Pipeline shutdown completed")
    
    def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        self.background_tasks = [
            asyncio.create_task(self._match_scraping_loop()),
            asyncio.create_task(self._odds_scraping_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._metrics_reporting_loop())
        ]
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def _match_scraping_loop(self):
        """Background task for continuous match scraping"""
        while self.is_running:
            try:
                await self._scrape_and_process_matches()
                await asyncio.sleep(self.config.match_scrape_interval)
            except Exception as e:
                logger.error(f"Error in match scraping loop: {e}")
                self.metrics.errors_encountered += 1
                await asyncio.sleep(60)
    
    async def _odds_scraping_loop(self):
        """Background task for continuous odds scraping"""
        while self.is_running:
            try:
                await self._scrape_odds_data()
                await asyncio.sleep(self.config.odds_scrape_interval)
            except Exception as e:
                logger.error(f"Error in odds scraping loop: {e}")
                self.metrics.errors_encountered += 1
                await asyncio.sleep(60)
    
    async def _health_monitoring_loop(self):
        """Background task for system health monitoring"""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_reporting_loop(self):
        """Background task for metrics reporting"""
        while self.is_running:
            try:
                await self._report_metrics()
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Error in metrics reporting: {e}")
                await asyncio.sleep(300)
    
    async def _scrape_and_process_matches(self):
        """Main pipeline: Scrape matches and process them end-to-end"""
        start_time = datetime.utcnow()
        
        try:
            # 1. Scrape upcoming matches with team stats
            logger.debug("Scraping upcoming matches...")
            matches = await self.hltv_scraper.get_upcoming_matches_with_stats(
                limit=self.config.max_concurrent_matches
            )
            
            if not matches:
                logger.debug("No upcoming matches found")
                return
            
            # 2. Process each match through the complete pipeline
            processed_count = 0
            for match_context in matches:
                try:
                    if match_context.match_id in self.processed_matches:
                        continue
                    
                    # Complete end-to-end processing
                    signals = await self._process_match_end_to_end(match_context)
                    
                    if signals:
                        processed_count += 1
                        self.processed_matches.add(match_context.match_id)
                        logger.info(f"Generated {len(signals)} signals for match {match_context.match_id}")
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing match {match_context.match_id}: {e}")
                    self.metrics.errors_encountered += 1
                    continue
            
            # Update metrics
            self.metrics.matches_processed += processed_count
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.average_processing_time = (
                self.metrics.average_processing_time * 0.9 + processing_time * 0.1
            )
            self.metrics.last_successful_run = datetime.utcnow()
            
            if processed_count > 0:
                logger.info(f"Processed {processed_count} matches in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in match scraping and processing: {e}")
            self.metrics.errors_encountered += 1
            raise
    
    async def _process_match_end_to_end(self, match_context: MatchContext) -> List[Any]:
        """Complete end-to-end processing: Match → Features → Prediction → Signal → Redis"""
        try:
            # 1. Get odds data for the match
            odds_data = await self._get_match_odds(match_context)
            if not odds_data:
                return []
            
            # 2. Extract features from match context
            features = await self._extract_match_features(match_context, odds_data)
            if features is None:
                return []
            
            # 3. Make prediction using ML model
            prediction_data = await self._make_prediction(match_context, features)
            if not prediction_data:
                return []
            
            # 4. Generate betting signals
            signals = await self._generate_signals(match_context, prediction_data, odds_data)
            if not signals:
                return []
            
            # 5. Publish signals to Redis
            published_signals = await self._publish_signals(signals)
            
            # 6. Store data in Redis schema
            await self._store_match_data(match_context, prediction_data, odds_data, signals)
            
            # Update metrics
            self.metrics.predictions_made += 1
            self.metrics.signals_generated += len(published_signals)
            
            return published_signals
            
        except Exception as e:
            logger.error(f"Error in end-to-end processing for {match_context.match_id}: {e}")
            raise
    
    async def _get_match_odds(self, match_context: MatchContext) -> Optional[Dict[str, Any]]:
        """Get odds data for a specific match"""
        try:
            # Get consensus odds for the match
            consensus = await self.odds_scraper.get_odds_for_match(
                match_context.team1_name, 
                match_context.team2_name
            )
            
            if not consensus:
                return None
            
            # Convert to standard format
            odds_data = {
                'match_winner': consensus.consensus_odds,
                'best_odds': consensus.best_odds,
                'bookmaker_count': consensus.bookmaker_count,
                'arbitrage_opportunity': consensus.arbitrage_opportunity,
                'arbitrage_profit': consensus.arbitrage_profit,
                'sources': consensus.sources,
                'timestamp': consensus.timestamp
            }
            
            return odds_data
            
        except Exception as e:
            logger.error(f"Error getting odds for match: {e}")
            return None
    
    async def _extract_match_features(
        self, 
        match_context: MatchContext, 
        odds_data: Dict[str, Any]
    ) -> Optional[List[float]]:
        """Extract features from match context and odds"""
        try:
            # Prepare data for feature engineering
            match_data = {
                'team1_name': match_context.team1_name,
                'team2_name': match_context.team2_name,
                'team1_ranking': match_context.team1_stats.world_ranking,
                'team2_ranking': match_context.team2_stats.world_ranking,
                'team1_form': match_context.team1_stats.win_rate_30d,
                'team2_form': match_context.team2_stats.win_rate_30d,
                'team1_map_winrates': match_context.team1_stats.map_stats,
                'team2_map_winrates': match_context.team2_stats.map_stats,
                'h2h_team1_wins': match_context.team1_stats.h2h_wins,
                'h2h_team2_wins': match_context.team2_stats.h2h_wins,
                'h2h_total': match_context.team1_stats.h2h_wins + match_context.team2_stats.h2h_wins,
                'event_name': match_context.event_name,
                'bo_format': match_context.bo_format,
                'team1_odds': odds_data['match_winner'].get(match_context.team1_name.lower(), 2.0),
                'team2_odds': odds_data['match_winner'].get(match_context.team2_name.lower(), 2.0),
                'bookmaker_count': odds_data['bookmaker_count']
            }
            
            # Extract features using feature engineer
            features = self.feature_engineer.extract_features(match_data)
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    async def _make_prediction(
        self, 
        match_context: MatchContext, 
        features: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Make prediction using the ML model"""
        try:
            if not self.prediction_model:
                return None
            
            # Make prediction
            prediction_result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.prediction_model.predict_match_outcome,
                {
                    'team1_name': match_context.team1_name,
                    'team2_name': match_context.team2_name,
                    'features': features
                }
            )
            
            if not prediction_result:
                return None
            
            # Check confidence threshold
            confidence = prediction_result.get('confidence', 0.0)
            if confidence < self.config.min_confidence_threshold:
                return None
            
            return {
                'predicted_winner': prediction_result.get('predicted_winner'),
                'confidence': confidence,
                'probabilities': prediction_result.get('probabilities', {}),
                'model_version': prediction_result.get('model_version', '1.0'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    async def _generate_signals(
        self, 
        match_context: MatchContext, 
        prediction_data: Dict[str, Any], 
        odds_data: Dict[str, Any]
    ) -> List[Any]:
        """Generate betting signals from prediction and odds"""
        try:
            # Prepare match data for signal generation
            match_data = {
                'match_id': match_context.match_id,
                'team1_name': match_context.team1_name,
                'team2_name': match_context.team2_name,
                'event_name': match_context.event_name,
                'match_time': match_context.match_time.isoformat() if match_context.match_time else None,
                'bo_format': match_context.bo_format
            }
            
            # Generate signals using the production signal generator
            signals = await self.signal_generator.generate_signals(
                match_data, 
                prediction_data, 
                odds_data
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _publish_signals(self, signals: List[Any]) -> List[Any]:
        """Publish signals to Redis channels"""
        published_signals = []
        
        for signal in signals:
            try:
                # Convert signal to dict for publishing
                signal_data = asdict(signal) if hasattr(signal, '__dict__') else signal
                
                # Publish to multiple channels
                await asyncio.gather(
                    self.redis_publisher.publish_signal(signal_data),
                    self.redis_schema.store_signal(signal_data),
                    return_exceptions=True
                )
                
                published_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error publishing signal: {e}")
                continue
        
        return published_signals
    
    async def _store_match_data(
        self, 
        match_context: MatchContext, 
        prediction_data: Dict[str, Any], 
        odds_data: Dict[str, Any], 
        signals: List[Any]
    ):
        """Store all match-related data in Redis"""
        try:
            # Store match data
            match_data = {
                'match_id': match_context.match_id,
                'team1_name': match_context.team1_name,
                'team2_name': match_context.team2_name,
                'team1_stats': asdict(match_context.team1_stats),
                'team2_stats': asdict(match_context.team2_stats),
                'event_name': match_context.event_name,
                'match_time': match_context.match_time.isoformat() if match_context.match_time else None,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await asyncio.gather(
                self.redis_schema.store_match(match_data),
                self.redis_schema.store_prediction(prediction_data),
                self.redis_schema.store_odds(odds_data),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Error storing match data: {e}")
    
    async def _scrape_odds_data(self):
        """Scrape odds from all sources and update cache"""
        try:
            consensus_list = await self.odds_scraper.scrape_all_sources()
            
            # Update odds cache
            for consensus in consensus_list:
                cache_key = f"consensus_{consensus.match_id}"
                self.odds_cache[cache_key] = (consensus, datetime.utcnow())
            
            logger.debug(f"Updated odds cache with {len(consensus_list)} consensus odds")
            
        except Exception as e:
            logger.error(f"Error scraping odds data: {e}")
    
    async def _train_initial_model(self):
        """Train initial model if none exists"""
        try:
            logger.info("Training initial ML model...")
            
            # Collect training data
            from ml_pipeline.training.data_collector import MLDataCollector
            collector = MLDataCollector()
            await collector.collect_historical_data(days_back=30, max_matches=500)
            
            # Train model
            results = self.model_trainer.train_all_models()
            
            # Save model
            if self.model_trainer.save_model(self.config.model_path):
                self.last_model_training = datetime.utcnow()
                logger.info("Initial model training completed")
            else:
                raise RuntimeError("Failed to save trained model")
                
        except Exception as e:
            logger.error(f"Error training initial model: {e}")
            raise
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks"""
        try:
            health_status = {
                'pipeline_running': self.is_running,
                'last_successful_run': self.metrics.last_successful_run,
                'matches_processed': self.metrics.matches_processed,
                'signals_generated': self.metrics.signals_generated,
                'error_rate': self.metrics.errors_encountered / max(1, self.metrics.matches_processed),
                'uptime': (datetime.utcnow() - self.metrics.uptime_start).total_seconds(),
                'cache_sizes': {
                    'team_stats': len(self.team_stats_cache),
                    'odds': len(self.odds_cache),
                    'features': len(self.feature_cache)
                }
            }
            
            # Store health status in Redis
            await self.redis_schema.store_health_status(health_status)
            
            # Check for alerts
            if health_status['error_rate'] > 0.1:  # 10% error rate
                await self.alert_system.send_alert(
                    "warning",
                    "High Error Rate",
                    f"Pipeline error rate: {health_status['error_rate']:.1%}",
                    health_status
                )
            
        except Exception as e:
            logger.error(f"Error in health checks: {e}")
    
    async def _report_metrics(self):
        """Report pipeline metrics to monitoring systems"""
        try:
            # Update Prometheus metrics
            self.prometheus_metrics.matches_processed_total.set(self.metrics.matches_processed)
            self.prometheus_metrics.signals_generated_total.set(self.metrics.signals_generated)
            self.prometheus_metrics.predictions_made_total.set(self.metrics.predictions_made)
            
            # Store metrics in Redis
            metrics_data = asdict(self.metrics)
            metrics_data['timestamp'] = datetime.utcnow().isoformat()
            await self.redis_schema.store_metrics(metrics_data)
            
        except Exception as e:
            logger.error(f"Error reporting metrics: {e}")
    
    async def run(self):
        """Run the integrated pipeline"""
        logger.info("Starting integrated CS2 betting pipeline...")
        
        try:
            await self.initialize()
            self.is_running = True
            
            # Send startup alert
            await self.alert_system.send_alert(
                "info",
                "Pipeline Started",
                "CS2 Betting Pipeline is now running",
                {"timestamp": datetime.utcnow().isoformat()}
            )
            
            # Keep running until stopped
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.alert_system.send_alert(
                "critical",
                "Pipeline Error",
                f"Pipeline encountered critical error: {str(e)}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
        finally:
            await self.close()
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'is_running': self.is_running,
            'metrics': asdict(self.metrics),
            'config': asdict(self.config),
            'cache_status': {
                'team_stats_cache_size': len(self.team_stats_cache),
                'odds_cache_size': len(self.odds_cache),
                'feature_cache_size': len(self.feature_cache)
            }
        }


# Usage example
async def main():
    """Main entry point for integrated pipeline"""
    pipeline = IntegratedPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
