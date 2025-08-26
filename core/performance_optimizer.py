#!/usr/bin/env python3
"""
Performance Optimizer for CS2 Betting Pipeline
Advanced parameter tuning, caching, and performance monitoring
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization"""
    avg_processing_time: float = 0.0
    peak_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    throughput_per_minute: float = 0.0
    redis_latency_ms: float = 0.0
    scraper_success_rate: float = 0.0
    prediction_accuracy: float = 0.0
    signal_quality_score: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    # Cache settings
    max_cache_size: int = 10000
    cache_ttl_seconds: int = 300
    cache_cleanup_interval: int = 600
    
    # Concurrency settings
    max_concurrent_scrapers: int = 5
    max_concurrent_predictions: int = 3
    thread_pool_size: int = 10
    
    # Rate limiting
    requests_per_second: float = 10.0
    burst_limit: int = 50
    
    # Memory management
    max_memory_mb: int = 1024
    gc_threshold: int = 1000
    
    # Performance targets
    target_processing_time_ms: float = 2000.0
    target_cache_hit_rate: float = 0.8
    target_error_rate: float = 0.05
    
    # Auto-tuning
    enable_auto_tuning: bool = True
    tuning_interval_minutes: int = 30
    performance_window_minutes: int = 60


class PerformanceOptimizer:
    """Advanced performance optimizer for the betting pipeline"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.metrics = PerformanceMetrics()
        self.historical_metrics = []
        
        # Performance tracking
        self.processing_times = []
        self.memory_samples = []
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.error_counts = {'total': 0, 'by_type': {}}
        
        # Optimization state
        self.current_parameters = {}
        self.optimization_history = []
        self.last_optimization = None
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # Rate limiting
        self.request_timestamps = []
        
        # Auto-tuning
        self.auto_tuning_enabled = self.config.enable_auto_tuning
        self.tuning_task = None
    
    async def initialize(self):
        """Initialize performance optimizer"""
        logger.info("Initializing performance optimizer...")
        
        # Set initial parameters
        self.current_parameters = {
            'scraper_delay': 1.0,
            'batch_size': 5,
            'cache_size': 1000,
            'concurrent_limit': 3,
            'retry_attempts': 3,
            'timeout_seconds': 30.0
        }
        
        # Start auto-tuning if enabled
        if self.auto_tuning_enabled:
            self.tuning_task = asyncio.create_task(self._auto_tuning_loop())
        
        logger.info("Performance optimizer initialized")
    
    async def close(self):
        """Cleanup optimizer resources"""
        if self.tuning_task:
            self.tuning_task.cancel()
        
        self.thread_pool.shutdown(wait=True)
        logger.info("Performance optimizer closed")
    
    def record_processing_time(self, duration_ms: float):
        """Record processing time for optimization"""
        self.processing_times.append(duration_ms)
        
        # Keep only recent samples
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-500:]
        
        # Update metrics
        self.metrics.avg_processing_time = statistics.mean(self.processing_times)
        self.metrics.peak_processing_time = max(self.processing_times)
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss for optimization"""
        if hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
        
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        self.metrics.cache_hit_rate = self.cache_stats['hits'] / max(1, total)
    
    def record_error(self, error_type: str):
        """Record error for optimization"""
        self.error_counts['total'] += 1
        self.error_counts['by_type'][error_type] = (
            self.error_counts['by_type'].get(error_type, 0) + 1
        )
    
    def record_memory_usage(self):
        """Record current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        self.memory_samples.append(memory_mb)
        if len(self.memory_samples) > 100:
            self.memory_samples = self.memory_samples[-50:]
        
        self.metrics.memory_usage_mb = statistics.mean(self.memory_samples)
        
        # Trigger GC if memory usage is high
        if memory_mb > self.config.max_memory_mb:
            gc.collect()
            logger.warning(f"High memory usage: {memory_mb:.1f}MB, triggered GC")
    
    def record_cpu_usage(self):
        """Record current CPU usage"""
        self.metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
    
    async def optimize_scraper_parameters(self, scraper_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize scraper parameters based on performance"""
        current_delay = self.current_parameters.get('scraper_delay', 1.0)
        success_rate = scraper_performance.get('success_rate', 0.0)
        avg_response_time = scraper_performance.get('avg_response_time', 1000.0)
        
        # Adjust delay based on success rate and response time
        if success_rate < 0.8:  # Low success rate
            new_delay = min(current_delay * 1.5, 5.0)  # Increase delay
        elif success_rate > 0.95 and avg_response_time < 500:  # High success, fast response
            new_delay = max(current_delay * 0.8, 0.1)  # Decrease delay
        else:
            new_delay = current_delay
        
        # Optimize concurrent requests
        current_concurrent = self.current_parameters.get('concurrent_limit', 3)
        if success_rate > 0.9 and self.metrics.cpu_usage_percent < 70:
            new_concurrent = min(current_concurrent + 1, self.config.max_concurrent_scrapers)
        elif success_rate < 0.7 or self.metrics.cpu_usage_percent > 90:
            new_concurrent = max(current_concurrent - 1, 1)
        else:
            new_concurrent = current_concurrent
        
        optimized_params = {
            'scraper_delay': new_delay,
            'concurrent_limit': new_concurrent,
            'batch_size': self._optimize_batch_size(),
            'timeout_seconds': self._optimize_timeout(),
            'retry_attempts': self._optimize_retry_attempts(success_rate)
        }
        
        # Update current parameters
        self.current_parameters.update(optimized_params)
        
        logger.info(f"Optimized scraper parameters: {optimized_params}")
        return optimized_params
    
    def _optimize_batch_size(self) -> int:
        """Optimize batch processing size"""
        if self.metrics.memory_usage_mb > self.config.max_memory_mb * 0.8:
            return max(self.current_parameters.get('batch_size', 5) - 1, 1)
        elif self.metrics.memory_usage_mb < self.config.max_memory_mb * 0.5:
            return min(self.current_parameters.get('batch_size', 5) + 1, 20)
        else:
            return self.current_parameters.get('batch_size', 5)
    
    def _optimize_timeout(self) -> float:
        """Optimize request timeout"""
        if self.metrics.avg_processing_time > self.config.target_processing_time_ms:
            return min(self.current_parameters.get('timeout_seconds', 30.0) + 5, 60.0)
        else:
            return max(self.current_parameters.get('timeout_seconds', 30.0) - 2, 10.0)
    
    def _optimize_retry_attempts(self, success_rate: float) -> int:
        """Optimize retry attempts based on success rate"""
        if success_rate < 0.6:
            return min(self.current_parameters.get('retry_attempts', 3) + 1, 5)
        elif success_rate > 0.95:
            return max(self.current_parameters.get('retry_attempts', 3) - 1, 1)
        else:
            return self.current_parameters.get('retry_attempts', 3)
    
    async def optimize_cache_parameters(self, cache_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cache parameters"""
        hit_rate = cache_performance.get('hit_rate', 0.0)
        cache_size = cache_performance.get('current_size', 0)
        memory_usage = cache_performance.get('memory_usage_mb', 0.0)
        
        # Optimize cache size
        if hit_rate < self.config.target_cache_hit_rate and memory_usage < self.config.max_memory_mb * 0.6:
            new_cache_size = min(cache_size * 1.2, self.config.max_cache_size)
        elif memory_usage > self.config.max_memory_mb * 0.8:
            new_cache_size = max(cache_size * 0.8, 100)
        else:
            new_cache_size = cache_size
        
        # Optimize TTL
        current_ttl = self.current_parameters.get('cache_ttl', 300)
        if hit_rate > 0.9:
            new_ttl = min(current_ttl * 1.1, 3600)  # Increase TTL
        elif hit_rate < 0.5:
            new_ttl = max(current_ttl * 0.9, 60)   # Decrease TTL
        else:
            new_ttl = current_ttl
        
        optimized_params = {
            'cache_size': int(new_cache_size),
            'cache_ttl': int(new_ttl),
            'cleanup_interval': self._optimize_cleanup_interval()
        }
        
        self.current_parameters.update(optimized_params)
        
        logger.info(f"Optimized cache parameters: {optimized_params}")
        return optimized_params
    
    def _optimize_cleanup_interval(self) -> int:
        """Optimize cache cleanup interval"""
        if self.metrics.memory_usage_mb > self.config.max_memory_mb * 0.7:
            return max(self.current_parameters.get('cleanup_interval', 600) // 2, 60)
        else:
            return min(self.current_parameters.get('cleanup_interval', 600) * 1.2, 1800)
    
    async def optimize_prediction_parameters(self, prediction_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize ML prediction parameters"""
        accuracy = prediction_performance.get('accuracy', 0.0)
        avg_inference_time = prediction_performance.get('avg_inference_time_ms', 100.0)
        confidence_distribution = prediction_performance.get('confidence_distribution', {})
        
        # Optimize confidence threshold
        current_threshold = self.current_parameters.get('confidence_threshold', 0.65)
        if accuracy < 0.6:
            new_threshold = min(current_threshold + 0.05, 0.9)  # Increase threshold
        elif accuracy > 0.8 and avg_inference_time < 50:
            new_threshold = max(current_threshold - 0.02, 0.5)  # Decrease threshold
        else:
            new_threshold = current_threshold
        
        # Optimize batch prediction size
        if avg_inference_time > 200:
            batch_size = max(self.current_parameters.get('prediction_batch_size', 10) - 2, 1)
        elif avg_inference_time < 50:
            batch_size = min(self.current_parameters.get('prediction_batch_size', 10) + 2, 50)
        else:
            batch_size = self.current_parameters.get('prediction_batch_size', 10)
        
        optimized_params = {
            'confidence_threshold': new_threshold,
            'prediction_batch_size': batch_size,
            'model_cache_size': self._optimize_model_cache_size(),
            'feature_cache_size': self._optimize_feature_cache_size()
        }
        
        self.current_parameters.update(optimized_params)
        
        logger.info(f"Optimized prediction parameters: {optimized_params}")
        return optimized_params
    
    def _optimize_model_cache_size(self) -> int:
        """Optimize model cache size"""
        if self.metrics.memory_usage_mb > self.config.max_memory_mb * 0.8:
            return max(self.current_parameters.get('model_cache_size', 100) - 20, 10)
        else:
            return min(self.current_parameters.get('model_cache_size', 100) + 10, 500)
    
    def _optimize_feature_cache_size(self) -> int:
        """Optimize feature cache size"""
        if self.metrics.cache_hit_rate > 0.9:
            return min(self.current_parameters.get('feature_cache_size', 1000) + 100, 5000)
        elif self.metrics.cache_hit_rate < 0.6:
            return max(self.current_parameters.get('feature_cache_size', 1000) - 100, 100)
        else:
            return self.current_parameters.get('feature_cache_size', 1000)
    
    async def check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        now = datetime.utcnow()
        
        # Remove old timestamps
        cutoff = now - timedelta(seconds=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        
        # Check rate limit
        if len(self.request_timestamps) >= self.config.requests_per_second:
            return False
        
        # Check burst limit
        burst_cutoff = now - timedelta(seconds=10)
        recent_requests = [ts for ts in self.request_timestamps if ts > burst_cutoff]
        if len(recent_requests) >= self.config.burst_limit:
            return False
        
        # Add current request
        self.request_timestamps.append(now)
        return True
    
    async def adaptive_delay(self, operation_type: str, success: bool, response_time_ms: float):
        """Calculate adaptive delay based on performance"""
        base_delay = self.current_parameters.get('scraper_delay', 1.0)
        
        # Adjust based on success and response time
        if not success:
            delay = base_delay * 2.0  # Double delay on failure
        elif response_time_ms > 2000:
            delay = base_delay * 1.5  # Increase delay for slow responses
        elif response_time_ms < 200:
            delay = base_delay * 0.5  # Decrease delay for fast responses
        else:
            delay = base_delay
        
        # Apply jitter to avoid thundering herd
        jitter = np.random.uniform(0.8, 1.2)
        final_delay = delay * jitter
        
        await asyncio.sleep(final_delay)
    
    async def _auto_tuning_loop(self):
        """Background task for automatic parameter tuning"""
        while True:
            try:
                await asyncio.sleep(self.config.tuning_interval_minutes * 60)
                await self._perform_auto_tuning()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-tuning loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _perform_auto_tuning(self):
        """Perform automatic parameter tuning"""
        logger.info("Starting automatic parameter tuning...")
        
        # Collect current performance metrics
        current_performance = self._collect_performance_metrics()
        
        # Store historical data
        self.historical_metrics.append({
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': asdict(self.metrics),
            'parameters': self.current_parameters.copy()
        })
        
        # Keep only recent history
        if len(self.historical_metrics) > 100:
            self.historical_metrics = self.historical_metrics[-50:]
        
        # Analyze trends and optimize
        optimization_results = await self._analyze_and_optimize(current_performance)
        
        # Log optimization results
        logger.info(f"Auto-tuning completed: {optimization_results}")
        
        self.last_optimization = datetime.utcnow()
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        return {
            'processing_times': self.processing_times[-100:] if self.processing_times else [],
            'memory_usage': self.metrics.memory_usage_mb,
            'cpu_usage': self.metrics.cpu_usage_percent,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'error_rate': self.metrics.error_rate,
            'throughput': self.metrics.throughput_per_minute
        }
    
    async def _analyze_and_optimize(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance and optimize parameters"""
        optimizations = {}
        
        # Optimize based on processing time
        if current_performance.get('processing_times'):
            avg_time = statistics.mean(current_performance['processing_times'])
            if avg_time > self.config.target_processing_time_ms:
                optimizations['reduce_batch_size'] = True
                optimizations['increase_timeout'] = True
        
        # Optimize based on memory usage
        if current_performance['memory_usage'] > self.config.max_memory_mb * 0.8:
            optimizations['reduce_cache_size'] = True
            optimizations['trigger_gc'] = True
        
        # Optimize based on cache performance
        if current_performance['cache_hit_rate'] < self.config.target_cache_hit_rate:
            optimizations['increase_cache_size'] = True
            optimizations['adjust_ttl'] = True
        
        # Optimize based on error rate
        if current_performance['error_rate'] > self.config.target_error_rate:
            optimizations['increase_retry_attempts'] = True
            optimizations['increase_delays'] = True
        
        return optimizations
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        return {
            'current_metrics': asdict(self.metrics),
            'current_parameters': self.current_parameters,
            'optimization_history': self.optimization_history[-10:],
            'performance_trends': self._calculate_performance_trends(),
            'recommendations': self._generate_recommendations(),
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None
        }
    
    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends"""
        if len(self.historical_metrics) < 2:
            return {"status": "insufficient_data"}
        
        recent = self.historical_metrics[-5:]
        older = self.historical_metrics[-10:-5] if len(self.historical_metrics) >= 10 else []
        
        trends = {}
        
        if older:
            recent_avg_time = statistics.mean([m['metrics']['avg_processing_time'] for m in recent])
            older_avg_time = statistics.mean([m['metrics']['avg_processing_time'] for m in older])
            
            if recent_avg_time < older_avg_time * 0.9:
                trends['processing_time'] = 'improving'
            elif recent_avg_time > older_avg_time * 1.1:
                trends['processing_time'] = 'degrading'
            else:
                trends['processing_time'] = 'stable'
        
        return trends
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if self.metrics.avg_processing_time > self.config.target_processing_time_ms:
            recommendations.append("Consider reducing batch sizes or increasing concurrency limits")
        
        if self.metrics.cache_hit_rate < self.config.target_cache_hit_rate:
            recommendations.append("Increase cache size or adjust TTL settings")
        
        if self.metrics.memory_usage_mb > self.config.max_memory_mb * 0.8:
            recommendations.append("Reduce cache sizes or implement more aggressive cleanup")
        
        if self.metrics.error_rate > self.config.target_error_rate:
            recommendations.append("Increase retry attempts or request delays")
        
        return recommendations


# Usage example
async def main():
    """Example usage of performance optimizer"""
    optimizer = PerformanceOptimizer()
    await optimizer.initialize()
    
    try:
        # Simulate some operations
        for i in range(100):
            start_time = datetime.utcnow()
            
            # Simulate processing
            await asyncio.sleep(0.1)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            optimizer.record_processing_time(processing_time)
            optimizer.record_cache_hit(i % 3 == 0)  # 33% hit rate
            optimizer.record_memory_usage()
            optimizer.record_cpu_usage()
        
        # Generate report
        report = optimizer.get_optimization_report()
        print(json.dumps(report, indent=2, default=str))
        
    finally:
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())
