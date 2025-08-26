import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import hashlib

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Redis key types with TTL specifications"""
    PREDICTION = "prediction"
    SIGNAL = "signal"
    MATCH = "match"
    ODDS = "odds"
    METRICS = "metrics"
    QUEUE = "queue"
    HEALTH = "health"
    CACHE = "cache"
    SESSION = "session"


@dataclass
class RedisKeyConfig:
    """Configuration for Redis key patterns and TTLs"""
    pattern: str
    ttl_seconds: int
    description: str
    indexed_fields: List[str] = None


class RedisSchemaManager:
    """Manages Redis key patterns, TTLs, and indexing"""
    
    # Key patterns and configurations
    KEY_CONFIGS = {
        # Predictions
        KeyType.PREDICTION: {
            'current': RedisKeyConfig(
                pattern="prediction:{match_id}",
                ttl_seconds=3600,  # 1 hour
                description="Current prediction for a match",
                indexed_fields=['team1', 'team2', 'confidence', 'expected_value']
            ),
            'history': RedisKeyConfig(
                pattern="prediction:history:{match_id}",
                ttl_seconds=86400 * 7,  # 7 days
                description="Historical predictions for a match"
            ),
            'recent': RedisKeyConfig(
                pattern="predictions:recent",
                ttl_seconds=3600 * 6,  # 6 hours
                description="Recent predictions list (FIFO)"
            ),
            'by_team': RedisKeyConfig(
                pattern="predictions:team:{team_name}",
                ttl_seconds=3600 * 2,  # 2 hours
                description="Predictions for specific team"
            )
        },
        
        # Signals
        KeyType.SIGNAL: {
            'active': RedisKeyConfig(
                pattern="signal:active:{signal_id}",
                ttl_seconds=3600 * 4,  # 4 hours
                description="Active betting signal"
            ),
            'queue': RedisKeyConfig(
                pattern="signals:queue",
                ttl_seconds=3600 * 12,  # 12 hours
                description="Signal processing queue"
            ),
            'processed': RedisKeyConfig(
                pattern="signals:processed",
                ttl_seconds=86400 * 3,  # 3 days
                description="Processed signals history"
            ),
            'by_confidence': RedisKeyConfig(
                pattern="signals:confidence:{bucket}",
                ttl_seconds=3600 * 6,  # 6 hours
                description="Signals grouped by confidence level"
            )
        },
        
        # Matches
        KeyType.MATCH: {
            'live': RedisKeyConfig(
                pattern="match:live:{match_id}",
                ttl_seconds=3600 * 8,  # 8 hours
                description="Live match data"
            ),
            'upcoming': RedisKeyConfig(
                pattern="match:upcoming:{match_id}",
                ttl_seconds=3600 * 24,  # 24 hours
                description="Upcoming match data"
            ),
            'completed': RedisKeyConfig(
                pattern="match:completed:{match_id}",
                ttl_seconds=86400 * 30,  # 30 days
                description="Completed match results"
            ),
            'stats': RedisKeyConfig(
                pattern="match:stats:{match_id}",
                ttl_seconds=86400 * 7,  # 7 days
                description="Match statistics and team data"
            )
        },
        
        # Odds
        KeyType.ODDS: {
            'current': RedisKeyConfig(
                pattern="odds:{team1}:{team2}",
                ttl_seconds=300,  # 5 minutes
                description="Current odds for match"
            ),
            'history': RedisKeyConfig(
                pattern="odds:history:{match_id}",
                ttl_seconds=86400 * 14,  # 14 days
                description="Odds movement history"
            ),
            'by_source': RedisKeyConfig(
                pattern="odds:source:{source}:{match_id}",
                ttl_seconds=600,  # 10 minutes
                description="Odds from specific bookmaker"
            ),
            'aggregated': RedisKeyConfig(
                pattern="odds:aggregated:{match_id}",
                ttl_seconds=180,  # 3 minutes
                description="Aggregated odds from multiple sources"
            )
        },
        
        # Metrics
        KeyType.METRICS: {
            'scraper': RedisKeyConfig(
                pattern="metrics:scraper:{date}",
                ttl_seconds=86400 * 7,  # 7 days
                description="Daily scraper metrics"
            ),
            'predictions': RedisKeyConfig(
                pattern="metrics:predictions:{date}",
                ttl_seconds=86400 * 7,  # 7 days
                description="Daily prediction metrics"
            ),
            'signals': RedisKeyConfig(
                pattern="metrics:signals:{date}",
                ttl_seconds=86400 * 7,  # 7 days
                description="Daily signal metrics"
            ),
            'performance': RedisKeyConfig(
                pattern="metrics:performance:realtime",
                ttl_seconds=300,  # 5 minutes
                description="Real-time performance metrics"
            )
        },
        
        # Queues
        KeyType.QUEUE: {
            'high_priority': RedisKeyConfig(
                pattern="queue:high_priority",
                ttl_seconds=3600 * 2,  # 2 hours
                description="High priority processing queue"
            ),
            'normal': RedisKeyConfig(
                pattern="queue:normal",
                ttl_seconds=3600 * 4,  # 4 hours
                description="Normal priority processing queue"
            ),
            'failed': RedisKeyConfig(
                pattern="queue:failed",
                ttl_seconds=86400,  # 24 hours
                description="Failed items for retry"
            ),
            'dead_letter': RedisKeyConfig(
                pattern="queue:dead_letter",
                ttl_seconds=86400 * 7,  # 7 days
                description="Dead letter queue for analysis"
            )
        },
        
        # Health & Status
        KeyType.HEALTH: {
            'service': RedisKeyConfig(
                pattern="health:service:{service_name}",
                ttl_seconds=60,  # 1 minute
                description="Service health status"
            ),
            'system': RedisKeyConfig(
                pattern="health:system",
                ttl_seconds=30,  # 30 seconds
                description="Overall system health"
            ),
            'alerts': RedisKeyConfig(
                pattern="health:alerts",
                ttl_seconds=3600,  # 1 hour
                description="Active health alerts"
            )
        },
        
        # Cache
        KeyType.CACHE: {
            'team_stats': RedisKeyConfig(
                pattern="cache:team_stats:{team_name}",
                ttl_seconds=3600,  # 1 hour
                description="Cached team statistics"
            ),
            'h2h': RedisKeyConfig(
                pattern="cache:h2h:{team1}:{team2}",
                ttl_seconds=7200,  # 2 hours
                description="Head-to-head statistics cache"
            ),
            'api_response': RedisKeyConfig(
                pattern="cache:api:{endpoint_hash}",
                ttl_seconds=300,  # 5 minutes
                description="API response cache"
            )
        }
    }
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Setup Redis indexes for searchable fields"""
        try:
            # Create search indexes if RediSearch is available
            from redisearch import Client, TextField, NumericField, IndexDefinition
            
            # Predictions index
            pred_client = Client('predictions_idx', conn=self.redis)
            try:
                pred_client.create_index([
                    TextField('team1'),
                    TextField('team2'),
                    NumericField('confidence'),
                    NumericField('expected_value'),
                    NumericField('timestamp')
                ], definition=IndexDefinition(prefix=['prediction:']))
            except Exception:
                pass  # Index might already exist
            
            # Signals index
            signal_client = Client('signals_idx', conn=self.redis)
            try:
                signal_client.create_index([
                    TextField('match_id'),
                    TextField('team'),
                    NumericField('confidence'),
                    NumericField('expected_value'),
                    TextField('status')
                ], definition=IndexDefinition(prefix=['signal:']))
            except Exception:
                pass
                
        except ImportError:
            logger.warning("RediSearch not available, using basic Redis operations")
    
    def get_key(self, key_type: KeyType, subtype: str, **kwargs) -> str:
        """Generate Redis key from pattern"""
        config = self.KEY_CONFIGS[key_type][subtype]
        return config.pattern.format(**kwargs)
    
    def set_with_ttl(self, key_type: KeyType, subtype: str, value: Any, **kwargs) -> bool:
        """Set value with automatic TTL"""
        try:
            config = self.KEY_CONFIGS[key_type][subtype]
            key = config.pattern.format(**kwargs)
            
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            
            return self.redis.setex(key, config.ttl_seconds, value)
        except Exception as e:
            logger.error(f"Failed to set key {key_type}/{subtype}: {e}")
            return False
    
    def get_with_json(self, key_type: KeyType, subtype: str, **kwargs) -> Optional[Any]:
        """Get value and attempt JSON decode"""
        try:
            config = self.KEY_CONFIGS[key_type][subtype]
            key = config.pattern.format(**kwargs)
            
            value = self.redis.get(key)
            if value is None:
                return None
            
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error(f"Failed to get key {key_type}/{subtype}: {e}")
            return None
    
    def list_keys(self, key_type: KeyType, subtype: str, pattern_kwargs: Dict = None) -> List[str]:
        """List keys matching pattern"""
        try:
            config = self.KEY_CONFIGS[key_type][subtype]
            if pattern_kwargs:
                pattern = config.pattern.format(**pattern_kwargs)
            else:
                # Replace format placeholders with wildcards
                pattern = config.pattern
                for placeholder in ['{match_id}', '{team_name}', '{team1}', '{team2}', 
                                  '{signal_id}', '{source}', '{date}', '{service_name}',
                                  '{endpoint_hash}', '{bucket}']:
                    pattern = pattern.replace(placeholder, '*')
            
            return self.redis.keys(pattern)
        except Exception as e:
            logger.error(f"Failed to list keys {key_type}/{subtype}: {e}")
            return []
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired keys and return counts"""
        cleanup_stats = {}
        
        for key_type, subtypes in self.KEY_CONFIGS.items():
            cleanup_stats[key_type.value] = 0
            
            for subtype, config in subtypes.items():
                try:
                    # Get all keys for this pattern
                    pattern = config.pattern
                    for placeholder in ['{match_id}', '{team_name}', '{team1}', '{team2}', 
                                      '{signal_id}', '{source}', '{date}', '{service_name}',
                                      '{endpoint_hash}', '{bucket}']:
                        pattern = pattern.replace(placeholder, '*')
                    
                    keys = self.redis.keys(pattern)
                    expired_count = 0
                    
                    for key in keys:
                        ttl = self.redis.ttl(key)
                        if ttl == -1:  # No TTL set
                            self.redis.expire(key, config.ttl_seconds)
                        elif ttl == -2:  # Key doesn't exist
                            expired_count += 1
                    
                    cleanup_stats[key_type.value] += expired_count
                    
                except Exception as e:
                    logger.error(f"Cleanup failed for {key_type}/{subtype}: {e}")
        
        return cleanup_stats
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get Redis memory usage by key patterns"""
        try:
            info = self.redis.info('memory')
            total_memory = info.get('used_memory', 0)
            
            pattern_usage = {}
            
            for key_type, subtypes in self.KEY_CONFIGS.items():
                pattern_usage[key_type.value] = {
                    'key_count': 0,
                    'estimated_memory': 0
                }
                
                for subtype, config in subtypes.items():
                    keys = self.list_keys(key_type, subtype)
                    pattern_usage[key_type.value]['key_count'] += len(keys)
                    
                    # Estimate memory usage (rough approximation)
                    for key in keys[:10]:  # Sample first 10 keys
                        try:
                            memory = self.redis.memory_usage(key)
                            if memory:
                                pattern_usage[key_type.value]['estimated_memory'] += memory
                        except Exception:
                            pass
                    
                    # Extrapolate for all keys
                    if len(keys) > 10:
                        avg_memory = pattern_usage[key_type.value]['estimated_memory'] / min(10, len(keys))
                        pattern_usage[key_type.value]['estimated_memory'] = avg_memory * len(keys)
            
            return {
                'total_memory_bytes': total_memory,
                'total_memory_mb': total_memory / (1024 * 1024),
                'pattern_breakdown': pattern_usage,
                'key_count_total': sum(p['key_count'] for p in pattern_usage.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    def search_predictions(self, query: Dict[str, Any]) -> List[Dict]:
        """Search predictions with filters"""
        try:
            # Try RediSearch first
            from redisearch import Client
            client = Client('predictions_idx', conn=self.redis)
            
            search_query = "*"
            filters = []
            
            if 'team' in query:
                filters.append(f"@team1:{query['team']} | @team2:{query['team']}")
            
            if 'min_confidence' in query:
                filters.append(f"@confidence:[{query['min_confidence']} +inf]")
            
            if 'min_expected_value' in query:
                filters.append(f"@expected_value:[{query['min_expected_value']} +inf]")
            
            if filters:
                search_query = " ".join(filters)
            
            results = client.search(search_query)
            return [json.loads(doc.json) for doc in results.docs]
            
        except ImportError:
            # Fallback to manual search
            return self._manual_search_predictions(query)
    
    def _manual_search_predictions(self, query: Dict[str, Any]) -> List[Dict]:
        """Manual search when RediSearch is not available"""
        results = []
        
        # Get recent predictions
        recent_keys = self.redis.lrange('predictions:recent', 0, -1)
        
        for key_data in recent_keys:
            try:
                prediction = json.loads(key_data)
                
                # Apply filters
                if 'team' in query:
                    team = query['team'].lower()
                    if (team not in prediction.get('teams', [{}])[0].get('name', '').lower() and
                        team not in prediction.get('teams', [{}])[1].get('name', '').lower()):
                        continue
                
                if 'min_confidence' in query:
                    if prediction.get('prediction', {}).get('confidence', 0) < query['min_confidence']:
                        continue
                
                if 'min_expected_value' in query:
                    if prediction.get('prediction', {}).get('expected_value', 0) < query['min_expected_value']:
                        continue
                
                results.append(prediction)
                
            except Exception as e:
                logger.warning(f"Failed to parse prediction: {e}")
                continue
        
        return results
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all queues"""
        stats = {}
        
        for queue_name in ['high_priority', 'normal', 'failed', 'dead_letter']:
            try:
                key = self.get_key(KeyType.QUEUE, queue_name)
                length = self.redis.llen(key)
                
                # Get sample items for analysis
                sample_items = self.redis.lrange(key, 0, 4)
                sample_data = []
                
                for item in sample_items:
                    try:
                        sample_data.append(json.loads(item))
                    except Exception:
                        pass
                
                stats[queue_name] = {
                    'length': length,
                    'sample_items': sample_data,
                    'oldest_item_age': self._get_oldest_item_age(key) if length > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"Failed to get queue stats for {queue_name}: {e}")
                stats[queue_name] = {'length': 0, 'sample_items': [], 'oldest_item_age': 0}
        
        return stats
    
    def _get_oldest_item_age(self, queue_key: str) -> int:
        """Get age of oldest item in queue (seconds)"""
        try:
            oldest_item = self.redis.lindex(queue_key, -1)
            if oldest_item:
                data = json.loads(oldest_item)
                if 'timestamp' in data:
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    return int((datetime.now() - timestamp).total_seconds())
        except Exception:
            pass
        return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive Redis health check"""
        try:
            start_time = datetime.now()
            
            # Basic connectivity
            ping_result = self.redis.ping()
            ping_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Redis info
            info = self.redis.info()
            
            # Memory usage
            memory_info = self.get_memory_usage()
            
            # Queue stats
            queue_stats = self.get_queue_stats()
            
            # Key counts by type
            key_counts = {}
            for key_type in KeyType:
                key_counts[key_type.value] = len(self.list_keys(key_type, list(self.KEY_CONFIGS[key_type].keys())[0]))
            
            return {
                'status': 'healthy' if ping_result else 'unhealthy',
                'ping_ms': ping_time,
                'redis_info': {
                    'version': info.get('redis_version'),
                    'uptime_seconds': info.get('uptime_in_seconds'),
                    'connected_clients': info.get('connected_clients'),
                    'used_memory_human': info.get('used_memory_human'),
                    'keyspace_hits': info.get('keyspace_hits'),
                    'keyspace_misses': info.get('keyspace_misses')
                },
                'memory_usage': memory_info,
                'queue_stats': queue_stats,
                'key_counts': key_counts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
