import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, Info, Enum, start_http_server
from prometheus_client.core import CollectorRegistry
import psutil
import threading
import asyncio

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Comprehensive Prometheus metrics for CS2 betting system"""
    
    def __init__(self, service_name: str, port: int = 9090):
        self.service_name = service_name
        self.port = port
        self.registry = CollectorRegistry()
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent', 
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes', 
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent', 
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'service_info', 
            'Service information',
            registry=self.registry
        )
        
        self.service_status = Enum(
            'service_status',
            'Service status',
            states=['starting', 'running', 'stopping', 'stopped', 'error'],
            registry=self.registry
        )
        
        # Scraper metrics
        self.scrape_requests_total = Counter(
            'scrape_requests_total',
            'Total scrape requests',
            ['source', 'status'],
            registry=self.registry
        )
        
        self.scrape_duration_seconds = Histogram(
            'scrape_duration_seconds',
            'Time spent scraping',
            ['source'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.matches_scraped_total = Counter(
            'matches_scraped_total',
            'Total matches scraped',
            ['source', 'status'],
            registry=self.registry
        )
        
        self.odds_updates_total = Counter(
            'odds_updates_total',
            'Total odds updates',
            ['source', 'bookmaker'],
            registry=self.registry
        )
        
        self.scraper_errors_total = Counter(
            'scraper_errors_total',
            'Total scraper errors',
            ['source', 'error_type'],
            registry=self.registry
        )
        
        # Prediction metrics
        self.predictions_generated_total = Counter(
            'predictions_generated_total',
            'Total predictions generated',
            ['model_type', 'confidence_bucket'],
            registry=self.registry
        )
        
        self.prediction_accuracy = Gauge(
            'prediction_accuracy_percent',
            'Prediction accuracy percentage',
            ['model_type', 'time_window'],
            registry=self.registry
        )
        
        self.model_inference_duration_seconds = Histogram(
            'model_inference_duration_seconds',
            'Model inference time',
            ['model_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        self.feature_extraction_duration_seconds = Histogram(
            'feature_extraction_duration_seconds',
            'Feature extraction time',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # Signal metrics
        self.signals_generated_total = Counter(
            'signals_generated_total',
            'Total signals generated',
            ['signal_type', 'confidence_level'],
            registry=self.registry
        )
        
        self.signals_sent_total = Counter(
            'signals_sent_total',
            'Total signals sent',
            ['channel', 'priority'],
            registry=self.registry
        )
        
        self.signal_processing_duration_seconds = Histogram(
            'signal_processing_duration_seconds',
            'Signal processing time',
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.active_signals = Gauge(
            'active_signals_count',
            'Number of active signals',
            ['signal_type'],
            registry=self.registry
        )
        
        # Redis metrics
        self.redis_operations_total = Counter(
            'redis_operations_total',
            'Total Redis operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.redis_connection_pool_size = Gauge(
            'redis_connection_pool_size',
            'Redis connection pool size',
            registry=self.registry
        )
        
        self.redis_operation_duration_seconds = Histogram(
            'redis_operation_duration_seconds',
            'Redis operation duration',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            'queue_size',
            'Queue size',
            ['queue_name'],
            registry=self.registry
        )
        
        self.queue_processing_duration_seconds = Histogram(
            'queue_processing_duration_seconds',
            'Queue item processing time',
            ['queue_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.queue_items_processed_total = Counter(
            'queue_items_processed_total',
            'Total queue items processed',
            ['queue_name', 'status'],
            registry=self.registry
        )
        
        # Risk management metrics
        self.positions_opened_total = Counter(
            'positions_opened_total',
            'Total positions opened',
            ['market_type'],
            registry=self.registry
        )
        
        self.positions_closed_total = Counter(
            'positions_closed_total',
            'Total positions closed',
            ['market_type', 'result'],
            registry=self.registry
        )
        
        self.current_exposure = Gauge(
            'current_exposure_amount',
            'Current exposure amount',
            ['currency'],
            registry=self.registry
        )
        
        self.bankroll_amount = Gauge(
            'bankroll_amount',
            'Current bankroll amount',
            ['currency'],
            registry=self.registry
        )
        
        self.kelly_stake_amount = Histogram(
            'kelly_stake_amount',
            'Kelly stake amounts',
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self.registry
        )
        
        # Performance metrics
        self.hit_rate_percent = Gauge(
            'hit_rate_percent',
            'Hit rate percentage',
            ['time_window', 'market_type'],
            registry=self.registry
        )
        
        self.roi_percent = Gauge(
            'roi_percent',
            'Return on investment percentage',
            ['time_window', 'market_type'],
            registry=self.registry
        )
        
        self.profit_loss_amount = Gauge(
            'profit_loss_amount',
            'Profit/Loss amount',
            ['currency', 'time_window'],
            registry=self.registry
        )
        
        # Alert metrics
        self.alerts_triggered_total = Counter(
            'alerts_triggered_total',
            'Total alerts triggered',
            ['alert_type', 'severity'],
            registry=self.registry
        )
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Initialize service info
        self.service_info.info({
            'service_name': service_name,
            'version': '1.0.0',
            'start_time': datetime.now().isoformat()
        })
        
        self.service_status.state('starting')
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self):
        """Start background thread for system metrics collection"""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_cpu_usage.set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_memory_usage.set(memory.used)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    disk_percent = (disk.used / disk.total) * 100
                    self.system_disk_usage.set(disk_percent)
                    
                    time.sleep(10)  # Collect every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(30)  # Wait longer on error
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def start_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            self.service_status.state('running')
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            self.service_status.state('error')
    
    # Scraper metric helpers
    def record_scrape_request(self, source: str, status: str, duration: float):
        """Record scrape request metrics"""
        self.scrape_requests_total.labels(source=source, status=status).inc()
        self.scrape_duration_seconds.labels(source=source).observe(duration)
    
    def record_match_scraped(self, source: str, status: str):
        """Record match scraped"""
        self.matches_scraped_total.labels(source=source, status=status).inc()
    
    def record_odds_update(self, source: str, bookmaker: str):
        """Record odds update"""
        self.odds_updates_total.labels(source=source, bookmaker=bookmaker).inc()
    
    def record_scraper_error(self, source: str, error_type: str):
        """Record scraper error"""
        self.scraper_errors_total.labels(source=source, error_type=error_type).inc()
    
    # Prediction metric helpers
    def record_prediction(self, model_type: str, confidence: float, inference_time: float):
        """Record prediction metrics"""
        confidence_bucket = self._get_confidence_bucket(confidence)
        self.predictions_generated_total.labels(
            model_type=model_type, 
            confidence_bucket=confidence_bucket
        ).inc()
        self.model_inference_duration_seconds.labels(model_type=model_type).observe(inference_time)
    
    def record_feature_extraction(self, duration: float):
        """Record feature extraction time"""
        self.feature_extraction_duration_seconds.observe(duration)
    
    def update_prediction_accuracy(self, model_type: str, time_window: str, accuracy: float):
        """Update prediction accuracy"""
        self.prediction_accuracy.labels(model_type=model_type, time_window=time_window).set(accuracy)
    
    # Signal metric helpers
    def record_signal_generated(self, signal_type: str, confidence_level: str):
        """Record signal generation"""
        self.signals_generated_total.labels(
            signal_type=signal_type, 
            confidence_level=confidence_level
        ).inc()
    
    def record_signal_sent(self, channel: str, priority: str, processing_time: float):
        """Record signal sent"""
        self.signals_sent_total.labels(channel=channel, priority=priority).inc()
        self.signal_processing_duration_seconds.observe(processing_time)
    
    def update_active_signals(self, signal_type: str, count: int):
        """Update active signals count"""
        self.active_signals.labels(signal_type=signal_type).set(count)
    
    # Redis metric helpers
    def record_redis_operation(self, operation: str, status: str, duration: float):
        """Record Redis operation"""
        self.redis_operations_total.labels(operation=operation, status=status).inc()
        self.redis_operation_duration_seconds.labels(operation=operation).observe(duration)
    
    def update_redis_pool_size(self, size: int):
        """Update Redis connection pool size"""
        self.redis_connection_pool_size.set(size)
    
    # Queue metric helpers
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size"""
        self.queue_size.labels(queue_name=queue_name).set(size)
    
    def record_queue_processing(self, queue_name: str, status: str, duration: float):
        """Record queue item processing"""
        self.queue_items_processed_total.labels(queue_name=queue_name, status=status).inc()
        self.queue_processing_duration_seconds.labels(queue_name=queue_name).observe(duration)
    
    # Risk management metric helpers
    def record_position_opened(self, market_type: str):
        """Record position opened"""
        self.positions_opened_total.labels(market_type=market_type).inc()
    
    def record_position_closed(self, market_type: str, result: str):
        """Record position closed"""
        self.positions_closed_total.labels(market_type=market_type, result=result).inc()
    
    def update_exposure(self, currency: str, amount: float):
        """Update current exposure"""
        self.current_exposure.labels(currency=currency).set(amount)
    
    def update_bankroll(self, currency: str, amount: float):
        """Update bankroll amount"""
        self.bankroll_amount.labels(currency=currency).set(amount)
    
    def record_kelly_stake(self, amount: float):
        """Record Kelly stake amount"""
        self.kelly_stake_amount.observe(amount)
    
    # Performance metric helpers
    def update_hit_rate(self, time_window: str, market_type: str, rate: float):
        """Update hit rate"""
        self.hit_rate_percent.labels(time_window=time_window, market_type=market_type).set(rate)
    
    def update_roi(self, time_window: str, market_type: str, roi: float):
        """Update ROI"""
        self.roi_percent.labels(time_window=time_window, market_type=market_type).set(roi)
    
    def update_pnl(self, currency: str, time_window: str, amount: float):
        """Update P&L"""
        self.profit_loss_amount.labels(currency=currency, time_window=time_window).set(amount)
    
    # Alert metric helpers
    def record_alert(self, alert_type: str, severity: str):
        """Record alert triggered"""
        self.alerts_triggered_total.labels(alert_type=alert_type, severity=severity).inc()
    
    # HTTP metric helpers
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request"""
        self.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for metrics"""
        if confidence < 0.6:
            return "low"
        elif confidence < 0.75:
            return "medium"
        elif confidence < 0.9:
            return "high"
        else:
            return "very_high"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        try:
            from prometheus_client import generate_latest
            metrics_data = generate_latest(self.registry).decode('utf-8')
            
            # Parse key metrics
            lines = metrics_data.split('\n')
            summary = {
                'service_name': self.service_name,
                'timestamp': datetime.now().isoformat(),
                'metrics_count': len([l for l in lines if l and not l.startswith('#')]),
                'key_metrics': {}
            }
            
            # Extract some key metrics
            for line in lines:
                if 'scrape_requests_total' in line and not line.startswith('#'):
                    summary['key_metrics']['scrape_requests'] = line.split()[-1]
                elif 'predictions_generated_total' in line and not line.startswith('#'):
                    summary['key_metrics']['predictions_generated'] = line.split()[-1]
                elif 'signals_sent_total' in line and not line.startswith('#'):
                    summary['key_metrics']['signals_sent'] = line.split()[-1]
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {
                'service_name': self.service_name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


# Global metrics instances
_metrics_instances: Dict[str, PrometheusMetrics] = {}


def get_metrics(service_name: str, port: int = None) -> PrometheusMetrics:
    """Get or create metrics instance for service"""
    if service_name not in _metrics_instances:
        if port is None:
            # Default ports for different services
            port_map = {
                'scraper': 9091,
                'prediction_tracker': 9092,
                'api_gateway': 9093,
                'signal_generator': 9094,
                'risk_manager': 9095
            }
            port = port_map.get(service_name, 9090)
        
        _metrics_instances[service_name] = PrometheusMetrics(service_name, port)
    
    return _metrics_instances[service_name]


# Decorator for timing functions
def time_function(metrics: PrometheusMetrics, metric_name: str, labels: Dict[str, str] = None):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record timing based on metric type
                if hasattr(metrics, metric_name):
                    metric = getattr(metrics, metric_name)
                    if labels:
                        metric.labels(**labels).observe(duration)
                    else:
                        metric.observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record error timing if needed
                raise
        
        return wrapper
    return decorator
