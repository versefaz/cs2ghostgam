import redis
import asyncio
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge
import logging
import os

from cs2_betting_system.scrapers.odds_scraper_robust import (
    MultiSourceOddsFetcher, OddsRecord, OddsSource
)
from models.odds_history import OddsHistory, AggregatedOdds, OddsMovement

# Metrics
odds_fetch_counter = Counter('odds_fetch_total', 'Total odds fetch attempts', ['source', 'status'])
odds_fetch_duration = Histogram('odds_fetch_duration_seconds', 'Odds fetch duration', ['source'])
active_sources_gauge = Gauge('active_odds_sources', 'Number of active odds sources')
odds_confidence_gauge = Gauge('odds_confidence', 'Average confidence score', ['source'])


class RobustOddsManager:
    """จัดการการดึงราคาแบบครบวงจร"""
    
    def __init__(self, db_session, redis_client=None):
        self.db = db_session
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            from urllib.parse import urlparse
            u = urlparse(redis_url)
            self.redis = redis.Redis(host=u.hostname, port=u.port or 6379, decode_responses=True)
        else:
            self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.fetcher = MultiSourceOddsFetcher()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.cache_ttl = 60  # seconds
        self.min_sources = 2
        self.confidence_threshold = 0.7
        self.monitor_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Setup components"""
        await self.fetcher.initialize()
        self.monitor_task = asyncio.create_task(self._monitor_sources())
        
    async def get_odds(
        self,
        match_id: str,
        force_refresh: bool = False
    ) -> Dict:
        """ดึงราคาพร้อม caching และ fallback"""
        
        # Check cache first
        if not force_refresh:
            cached = self._get_cached_odds(match_id)
            if cached:
                return cached
        
        # Fetch from sources
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        
        try:
            # Get match identifiers for different sources
            match_identifiers = await self._resolve_match_identifiers(match_id)
            
            # Fetch from multiple sources
            odds_records = await self.fetcher.fetch_odds_with_fallback(
                match_identifiers,
                self.confidence_threshold
            )
            
            # Record metrics
            duration = loop.time() - start_time
            for record in odds_records:
                odds_fetch_counter.labels(source=record.source, status='success').inc()
                odds_fetch_duration.labels(source=record.source).observe(duration)
                odds_confidence_gauge.labels(source=record.source).set(record.confidence)
            
            # Store in database
            await self._store_odds_history(odds_records)
            
            # Aggregate odds
            aggregated = self.fetcher.aggregate_odds(odds_records)
            
            # Update aggregated table
            await self._update_aggregated_odds(match_id, aggregated)
            
            # Detect movement
            movement = await self._detect_movement(match_id, aggregated)
            
            # Cache results
            payload = {
                "match_id": match_id,
                "aggregated": aggregated,
                "records": [r.to_dict() for r in odds_records],
                "movement": movement,
                "metadata": {
                    "sources_count": len(odds_records),
                    "fetch_time": duration,
                    "timestamp": datetime.now().isoformat(),
                    "cache_ttl": self.cache_ttl
                }
            }
            self._cache_odds(match_id, payload)
            return payload
            
        except Exception as e:
            self.logger.error(f"Failed to fetch odds for {match_id}: {e}")
            odds_fetch_counter.labels(source='all', status='failure').inc()
            
            # Try to return cached data even if expired
            cached = self._get_cached_odds(match_id, ignore_ttl=True)
            if cached:
                cached["metadata"]["from_expired_cache"] = True
                return cached
            
            raise
    
    async def _resolve_match_identifiers(self, match_id: str) -> Dict[OddsSource, str]:
        """แปลง match_id เป็น URL/ID สำหรับแต่ละ source"""
        identifiers: Dict[OddsSource, str] = {}
        # Example implementation
        identifiers[OddsSource.ODDSPORTAL] = f"https://oddsportal.com/esports/match/{match_id}"
        identifiers[OddsSource.GGBET] = f"https://ggbet.com/match/{match_id}"
        identifiers[OddsSource.PINNACLE] = match_id  # Pinnacle might use same ID
        bet365 = await self._resolve_bet365_id(match_id)
        if bet365:
            identifiers[OddsSource.BET365] = bet365
        return identifiers
    
    async def _store_odds_history(self, records: List[OddsRecord]):
        """บันทึกประวัติราคา"""
        for record in records:
            history = OddsHistory(
                match_id=record.raw_data.get('match_id', ''),
                team1=record.team1,
                team2=record.team2,
                odds_1=record.odds_1,
                odds_2=record.odds_2,
                odds_draw=record.odds_draw,
                source=record.source,
                market_type=record.market_type,
                timestamp=record.timestamp,
                match_time=record.match_time,
                confidence=record.confidence,
                raw_data=record.raw_data
            )
            self.db.add(history)
        
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to store odds history: {e}")
    
    async def _detect_movement(self, match_id: str, current_odds: Dict) -> Dict:
        """ตรวจจับการเคลื่อนไหวราคา"""
        # Get historical odds
        history = self.db.query(OddsHistory).filter(
            OddsHistory.match_id == match_id
        ).order_by(OddsHistory.timestamp.desc()).limit(50).all()
        
        if len(history) < 2:
            return {"status": "insufficient_data"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'timestamp': h.timestamp,
            'odds_1': h.odds_1,
            'odds_2': h.odds_2,
            'source': h.source
        } for h in history])
        
        # Group by source and analyze
        movements: Dict[str, Dict] = {}
        for source in df['source'].unique():
            source_df = df[df['source'] == source].sort_values('timestamp')
            
            if len(source_df) >= 2:
                movements[source] = {
                    'odds_1_open': float(source_df['odds_1'].iloc[0]),
                    'odds_1_current': float(source_df['odds_1'].iloc[-1]),
                    'odds_1_change': float(source_df['odds_1'].iloc[-1] - source_df['odds_1'].iloc[0]),
                    'odds_1_pct': float(((source_df['odds_1'].iloc[-1] / source_df['odds_1'].iloc[0]) - 1) * 100),
                    'odds_2_open': float(source_df['odds_2'].iloc[0]),
                    'odds_2_current': float(source_df['odds_2'].iloc[-1]),
                    'odds_2_change': float(source_df['odds_2'].iloc[-1] - source_df['odds_2'].iloc[0]),
                    'odds_2_pct': float(((source_df['odds_2'].iloc[-1] / source_df['odds_2'].iloc[0]) - 1) * 100),
                    'trend': self._calculate_trend(source_df),
                    'volatility': float(source_df['odds_1'].std() or 0.0),
                    'samples': int(len(source_df))
                }
        
        # Overall movement analysis
        overall_movement = {
            'sources': movements,
            'summary': self._summarize_movement(movements),
            'alert': self._check_movement_alerts(movements)
        }
        
        # Store movement data
        await self._update_movement_table(match_id, overall_movement)
        
        return overall_movement
    
    def _calculate_trend(self, df: pd.DataFrame) -> str:
        """คำนวณ trend ของราคา"""
        if len(df) < 3:
            return "insufficient_data"
        
        # Simple linear regression on odds_1
        x = np.arange(len(df))
        y = df['odds_1'].values
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "rising"
        else:
            return "falling"
    
    def _summarize_movement(self, movements: Dict) -> Dict:
        """สรุปการเคลื่อนไหวราคา"""
        if not movements:
            return {"status": "no_data"}
        
        # Calculate averages across sources
        avg_change_1 = float(np.mean([m['odds_1_pct'] for m in movements.values()]))
        avg_change_2 = float(np.mean([m['odds_2_pct'] for m in movements.values()]))
        
        # Determine market sentiment
        if avg_change_1 > 5:
            sentiment = "strong_shift_to_team1"
        elif avg_change_1 < -5:
            sentiment = "strong_shift_to_team2"
        elif abs(avg_change_1) < 2 and abs(avg_change_2) < 2:
            sentiment = "stable"
        else:
            sentiment = "moderate_movement"
        
        return {
            "avg_change_team1": avg_change_1,
            "avg_change_team2": avg_change_2,
            "market_sentiment": sentiment,
            "sources_analyzed": len(movements),
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_movement_alerts(self, movements: Dict) -> List[Dict]:
        """ตรวจสอบ alerts สำหรับการเคลื่อนไหวผิดปกติ"""
        alerts: List[Dict] = []
        
        for source, data in movements.items():
            # Alert for large movements
            if abs(data['odds_1_pct']) > 10:
                alerts.append({
                    "type": "large_movement",
                    "source": source,
                    "team": "team1",
                    "change_pct": data['odds_1_pct'],
                    "severity": "high"
                })
            
            # Alert for high volatility
            if data['volatility'] and data['volatility'] > 0.5:
                alerts.append({
                    "type": "high_volatility",
                    "source": source,
                    "volatility": data['volatility'],
                    "severity": "medium"
                })
        
        return alerts
    
    async def _update_movement_table(self, match_id: str, movement_data: Dict):
        """อัปเดตตารางการเคลื่อนไหว"""
        for source, data in movement_data.get('sources', {}).items():
            # Check if record exists
            existing = self.db.query(OddsMovement).filter(
                OddsMovement.match_id == match_id,
                OddsMovement.source == source
            ).first()
            
            if existing:
                # Update existing
                existing.odds_1_current = data.get('odds_1_current')
                existing.odds_2_current = data.get('odds_2_current')
                existing.odds_1_change = data.get('odds_1_change')
                existing.odds_2_change = data.get('odds_2_change')
                existing.odds_1_change_pct = data.get('odds_1_pct')
                existing.odds_2_change_pct = data.get('odds_2_pct')
                existing.last_updated = datetime.now()
                existing.update_count = (existing.update_count or 0) + 1
                existing.trend_direction = data.get('trend')
                existing.volatility = data.get('volatility')
            else:
                # Create new
                movement = OddsMovement(
                    match_id=match_id,
                    source=source,
                    odds_1_open=data.get('odds_1_open'),
                    odds_1_current=data.get('odds_1_current'),
                    odds_1_change=data.get('odds_1_change'),
                    odds_1_change_pct=data.get('odds_1_pct'),
                    odds_2_open=data.get('odds_2_open'),
                    odds_2_current=data.get('odds_2_current'),
                    odds_2_change=data.get('odds_2_change'),
                    odds_2_change_pct=data.get('odds_2_pct'),
                    first_seen=datetime.now(),
                    last_updated=datetime.now(),
                    update_count=1,
                    trend_direction=data.get('trend'),
                    volatility=data.get('volatility')
                )
                self.db.add(movement)
        
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to update movement table: {e}")
    
    async def _update_aggregated_odds(self, match_id: str, aggregated: Dict):
        """อัปเดตตาราง aggregated odds"""
        if not aggregated:
            return
        
        for key, data in aggregated.items():
            # Calculate statistics
            implied_prob_1 = 1 / data['odds_1'] if data['odds_1'] and data['odds_1'] > 0 else 0
            implied_prob_2 = 1 / data['odds_2'] if data['odds_2'] and data['odds_2'] > 0 else 0
            market_margin = implied_prob_1 + implied_prob_2 - 1
            
            # Check if exists
            existing = self.db.query(AggregatedOdds).filter(
                AggregatedOdds.match_id == match_id
            ).first()
            
            if existing:
                existing.avg_odds_1 = data['odds_1']
                existing.avg_odds_2 = data['odds_2']
                existing.best_odds_1 = data['best_odds_1']
                existing.best_odds_2 = data['best_odds_2']
                existing.best_source_1 = data.get('best_source_1')
                existing.best_source_2 = data.get('best_source_2')
                existing.num_sources = data['num_sources']
                existing.sources = data['sources']
                existing.overall_confidence = data['confidence']
                existing.last_aggregation = datetime.now()
                existing.market_margin = market_margin
                existing.implied_prob_1 = implied_prob_1
                existing.implied_prob_2 = implied_prob_2
            else:
                agg = AggregatedOdds(
                    match_id=match_id,
                    avg_odds_1=data['odds_1'],
                    avg_odds_2=data['odds_2'],
                    best_odds_1=data['best_odds_1'],
                    best_odds_2=data['best_odds_2'],
                    best_source_1=data.get('best_source_1'),
                    best_source_2=data.get('best_source_2'),
                    num_sources=data['num_sources'],
                    sources=data['sources'],
                    overall_confidence=data['confidence'],
                    last_aggregation=datetime.now(),
                    market_margin=market_margin,
                    implied_prob_1=implied_prob_1,
                    implied_prob_2=implied_prob_2
                )
                self.db.add(agg)
        
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to update aggregated odds: {e}")
    
    def _get_cached_odds(self, match_id: str, ignore_ttl: bool = False) -> Optional[Dict]:
        """ดึงราคาจาก cache"""
        cache_key = f"odds:{match_id}"
        
        try:
            cached = self.redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                
                # Check TTL
                if not ignore_ttl:
                    ts = data.get('metadata', {}).get('timestamp')
                    if ts:
                        cached_time = datetime.fromisoformat(ts)
                        if datetime.now() - cached_time > timedelta(seconds=self.cache_ttl):
                            return None
                
                return data
        except Exception as e:
            self.logger.error(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_odds(self, match_id: str, data: Dict):
        """เก็บราคาใน cache"""
        cache_key = f"odds:{match_id}"
        
        try:
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            self.logger.error(f"Cache storage failed: {e}")
    
    async def _monitor_sources(self):
        """Monitor source health"""
        while True:
            try:
                # Check each source
                active_count = 0
                for source in self.fetcher.priority_sources:
                    # Simple health check
                    if await self._check_source_health(source):
                        active_count += 1
                
                active_sources_gauge.set(active_count)
                
                # Log if sources are down
                if active_count < self.min_sources:
                    self.logger.warning(f"Only {active_count} sources active!")
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _check_source_health(self, source: OddsSource) -> bool:
        """ตรวจสอบสุขภาพของ source"""
        # Example: Check recent success rate from DB
        recent_odds = self.db.query(OddsHistory).filter(
            OddsHistory.source == source.value if hasattr(source, 'value') else str(source),
            OddsHistory.timestamp > datetime.now() - timedelta(minutes=10)
        ).count()
        return recent_odds > 0
    
    async def _resolve_bet365_id(self, match_id: str) -> Optional[str]:
        """แปลง match_id สำหรับ bet365"""
        # Placeholder mapping logic
        return f"bet365_{match_id}"
    
    async def get_best_odds(self, match_id: str) -> Dict:
        """ดึงราคาที่ดีที่สุด"""
        # Get latest aggregated odds
        agg = self.db.query(AggregatedOdds).filter(
            AggregatedOdds.match_id == match_id
        ).order_by(AggregatedOdds.last_aggregation.desc()).first()
        
        if not agg:
            # Fetch fresh if not available
            await self.get_odds(match_id)
            agg = self.db.query(AggregatedOdds).filter(
                AggregatedOdds.match_id == match_id
            ).first()
        
        if agg:
            return {
                "team1": {
                    "best_odds": agg.best_odds_1,
                    "source": agg.best_source_1,
                    "implied_prob": agg.implied_prob_1
                },
                "team2": {
                    "best_odds": agg.best_odds_2,
                    "source": agg.best_source_2,
                    "implied_prob": agg.implied_prob_2
                },
                "market_margin": agg.market_margin,
                "num_sources": agg.num_sources,
                "confidence": agg.overall_confidence,
                "last_update": agg.last_aggregation.isoformat() if agg.last_aggregation else None
            }
        
        return {}
    
    async def get_odds_history(
        self,
        match_id: str,
        hours_back: int = 24
    ) -> pd.DataFrame:
        """ดึงประวัติราคา"""
        since = datetime.now() - timedelta(hours=hours_back)
        
        history = self.db.query(OddsHistory).filter(
            OddsHistory.match_id == match_id,
            OddsHistory.timestamp > since
        ).order_by(OddsHistory.timestamp).all()
        
        if history:
            df = pd.DataFrame([{
                'timestamp': h.timestamp,
                'team1': h.team1,
                'team2': h.team2,
                'odds_1': h.odds_1,
                'odds_2': h.odds_2,
                'source': h.source,
                'confidence': h.confidence
            } for h in history])
            
            return df
        
        return pd.DataFrame()
