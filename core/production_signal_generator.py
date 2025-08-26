#!/usr/bin/env python3
"""
Production Signal Generator - Unified Implementation
Consolidates all signal generation logic with Redis publishing, risk management, and monitoring
"""

import asyncio
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

# Core imports
from models.signal import BettingSignal, SignalSide, SignalPriority
from utils.kelly import calculate_kelly_fraction
from risk_management.kelly_calculator import KellyCalculator
from risk_management.cooldown_manager import CooldownManager
from monitoring.prometheus_metrics import PrometheusMetrics

# Redis imports with fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class SignalStrategy(Enum):
    VALUE_BETTING = "value_betting"
    ARBITRAGE = "arbitrage"
    ML_PREDICTION = "ml_prediction"
    PATTERN_ANALYSIS = "pattern_analysis"
    LIVE_EDGE = "live_edge"


@dataclass
class SignalConfig:
    """Comprehensive signal generation configuration"""
    # Core thresholds
    min_ev: float = 0.05  # 5% minimum expected value
    min_confidence: float = 0.65  # 65% minimum confidence
    max_stake: float = 0.10  # 10% maximum stake of bankroll
    kelly_multiplier: float = 0.25  # Quarter Kelly for safety
    
    # Strategy toggles
    enable_value_betting: bool = True
    enable_arbitrage: bool = True
    enable_ml_signals: bool = True
    enable_pattern_signals: bool = True
    enable_live_signals: bool = True
    
    # Risk management
    max_concurrent_signals: int = 5
    max_exposure: float = 0.30  # 30% total exposure
    max_daily_signals: int = 20
    signal_expiry_minutes: int = 5
    
    # Priority thresholds
    high_priority_ev: float = 0.15  # 15% EV for high priority
    critical_priority_ev: float = 0.25  # 25% EV for critical
    
    # Arbitrage settings
    min_arbitrage_profit: float = 0.02  # 2% minimum arb profit
    max_arbitrage_exposure: float = 0.50  # 50% for arb opportunities
    
    # ML model settings
    ml_confidence_threshold: float = 0.70
    ml_edge_multiplier: float = 1.2
    
    # Pattern analysis
    pattern_lookback_days: int = 30
    pattern_min_samples: int = 10


class ProductionSignalGenerator:
    """Production-ready unified signal generator"""
    
    def __init__(
        self, 
        config: SignalConfig = None, 
        redis_url: str = "redis://localhost:6379",
        bankroll: float = 10000.0
    ):
        self.config = config or SignalConfig()
        self.bankroll = bankroll
        self.active_signals: Dict[str, BettingSignal] = {}
        self.daily_signal_count = 0
        self.total_exposure = 0.0
        self._lock = asyncio.Lock()
        
        # Components
        self.kelly_calc = KellyCalculator()
        self.cooldown_mgr = CooldownManager()
        self.metrics = PrometheusMetrics()
        
        # Redis setup
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.publisher_id = str(uuid.uuid4())[:8]
        
        # Redis keys
        self.signal_queue = "cs2:signals:queue"
        self.signal_history = "cs2:signals:history"
        self.signal_active = "cs2:signals:active"
        self.signal_metrics = "cs2:signals:metrics"
        
        # Tracking
        self.processed_matches: Set[str] = set()
        self.last_reset = datetime.utcnow().date()
        
    async def initialize(self):
        """Initialize all components"""
        # Redis connection
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info(f"SignalGenerator connected to Redis: {self.redis_url}")
                
                # Load active signals from Redis
                await self._load_active_signals()
                
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Initialize components
        await self.cooldown_mgr.initialize()
        
        # Start background tasks
        asyncio.create_task(self._cleanup_expired_signals())
        asyncio.create_task(self._daily_reset_task())
        
        logger.info("ProductionSignalGenerator initialized successfully")
    
    async def close(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
        await self.cooldown_mgr.close()
    
    async def generate_signals(
        self, 
        match_data: Dict[str, Any], 
        prediction_data: Dict[str, Any] = None,
        odds_data: Dict[str, Any] = None
    ) -> List[BettingSignal]:
        """Main entry point for signal generation"""
        async with self._lock:
            try:
                match_id = match_data.get('match_id')
                if not match_id:
                    logger.warning("No match_id provided")
                    return []
                
                # Check daily limits
                if not await self._check_daily_limits():
                    return []
                
                # Check cooldown
                if await self.cooldown_mgr.is_match_in_cooldown(match_id):
                    logger.debug(f"Match {match_id} in cooldown")
                    return []
                
                # Check if already processed
                if match_id in self.processed_matches:
                    return []
                
                signals = []
                
                # Generate signals from different strategies
                if self.config.enable_value_betting and prediction_data and odds_data:
                    value_signals = await self._generate_value_signals(
                        match_data, prediction_data, odds_data
                    )
                    signals.extend(value_signals)
                
                if self.config.enable_arbitrage and odds_data:
                    arb_signals = await self._generate_arbitrage_signals(
                        match_data, odds_data
                    )
                    signals.extend(arb_signals)
                
                if self.config.enable_ml_signals and prediction_data:
                    ml_signals = await self._generate_ml_signals(
                        match_data, prediction_data, odds_data
                    )
                    signals.extend(ml_signals)
                
                if self.config.enable_pattern_signals:
                    pattern_signals = await self._generate_pattern_signals(
                        match_data, odds_data
                    )
                    signals.extend(pattern_signals)
                
                # Filter and validate signals
                valid_signals = await self._validate_signals(signals)
                
                # Apply risk management
                final_signals = await self._apply_risk_management(valid_signals)
                
                # Publish signals
                published_signals = []
                for signal in final_signals:
                    if await self._publish_signal(signal):
                        published_signals.append(signal)
                        await self._track_signal(signal)
                
                # Mark match as processed
                self.processed_matches.add(match_id)
                
                # Update metrics
                self.metrics.signals_generated_total.inc(len(published_signals))
                
                logger.info(f"Generated {len(published_signals)} signals for match {match_id}")
                return published_signals
                
            except Exception as e:
                logger.error(f"Error generating signals: {e}")
                self.metrics.signal_generation_errors_total.inc()
                return []
    
    async def _generate_value_signals(
        self, 
        match_data: Dict[str, Any], 
        prediction_data: Dict[str, Any],
        odds_data: Dict[str, Any]
    ) -> List[BettingSignal]:
        """Generate value betting signals"""
        signals = []
        
        try:
            match_id = match_data['match_id']
            our_probs = prediction_data.get('probabilities', {})
            market_odds = odds_data.get('match_winner', {})
            
            for side, our_prob in our_probs.items():
                if side not in market_odds:
                    continue
                
                odds = market_odds[side]
                implied_prob = 1.0 / odds
                
                # Calculate edge
                edge = our_prob - implied_prob
                ev = (our_prob * odds) - 1
                
                if ev < self.config.min_ev:
                    continue
                
                # Calculate confidence
                confidence = min(0.95, prediction_data.get('confidence', 0.5) + (edge * 0.5))
                
                if confidence < self.config.min_confidence:
                    continue
                
                # Calculate Kelly stake
                kelly = self.kelly_calc.calculate_kelly_fraction(
                    probability=our_prob,
                    odds=odds,
                    multiplier=self.config.kelly_multiplier
                )
                
                stake = min(kelly, self.config.max_stake)
                priority = self._determine_priority(ev)
                
                signal = BettingSignal(
                    match_id=match_id,
                    side=SignalSide(side.lower()),
                    stake=stake,
                    ev=ev,
                    confidence=confidence,
                    odds=odds,
                    probability=our_prob,
                    kelly_fraction=kelly,
                    source="value_betting",
                    strategy=SignalStrategy.VALUE_BETTING.value,
                    priority=priority,
                    expires_at=datetime.utcnow() + timedelta(minutes=self.config.signal_expiry_minutes),
                    reasons=[
                        f"Value edge: {edge:.2%}",
                        f"Expected value: {ev:.2%}",
                        f"Our prob: {our_prob:.2%} vs Market: {implied_prob:.2%}",
                        f"Kelly fraction: {kelly:.2%}"
                    ],
                    metadata={
                        "team1": match_data.get('team1_name'),
                        "team2": match_data.get('team2_name'),
                        "event": match_data.get('event_name'),
                        "match_time": match_data.get('match_time'),
                        "model_version": prediction_data.get('model_version'),
                        "edge": edge,
                        "implied_prob": implied_prob
                    }
                )
                
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"Error generating value signals: {e}")
        
        return signals
    
    async def _generate_arbitrage_signals(
        self, 
        match_data: Dict[str, Any], 
        odds_data: Dict[str, Any]
    ) -> List[BettingSignal]:
        """Generate arbitrage signals"""
        signals = []
        
        try:
            match_id = match_data['match_id']
            
            # Check for arbitrage opportunities across bookmakers
            for market, bookmaker_odds in odds_data.items():
                if not isinstance(bookmaker_odds, dict) or len(bookmaker_odds) < 2:
                    continue
                
                arb_opportunity = self._calculate_arbitrage(bookmaker_odds)
                
                if not arb_opportunity or arb_opportunity['profit'] < self.config.min_arbitrage_profit:
                    continue
                
                # Create signals for each side of the arbitrage
                for side, bet_info in arb_opportunity['bets'].items():
                    stake = min(bet_info['stake'], self.config.max_arbitrage_exposure)
                    
                    signal = BettingSignal(
                        match_id=match_id,
                        side=SignalSide(side.lower()),
                        stake=stake,
                        ev=arb_opportunity['profit'],
                        confidence=0.95,  # Arbitrage is nearly risk-free
                        odds=bet_info['odds'],
                        probability=1.0 / bet_info['odds'],
                        kelly_fraction=stake,
                        source=f"arbitrage_{bet_info['bookmaker']}",
                        strategy=SignalStrategy.ARBITRAGE.value,
                        priority=SignalPriority.HIGH,
                        expires_at=datetime.utcnow() + timedelta(minutes=2),  # Short expiry for arb
                        reasons=[
                            f"Arbitrage profit: {arb_opportunity['profit']:.2%}",
                            f"Bookmaker: {bet_info['bookmaker']}",
                            f"Total return guaranteed: {arb_opportunity['total_return']:.2%}"
                        ],
                        metadata={
                            "arbitrage_group": arb_opportunity['group_id'],
                            "bookmaker": bet_info['bookmaker'],
                            "market": market,
                            "total_stake_required": arb_opportunity['total_stake']
                        }
                    )
                    
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error generating arbitrage signals: {e}")
        
        return signals
    
    async def _generate_ml_signals(
        self, 
        match_data: Dict[str, Any], 
        prediction_data: Dict[str, Any],
        odds_data: Dict[str, Any] = None
    ) -> List[BettingSignal]:
        """Generate ML-based signals"""
        signals = []
        
        try:
            match_id = match_data['match_id']
            confidence = prediction_data.get('confidence', 0)
            
            if confidence < self.config.ml_confidence_threshold:
                return signals
            
            predicted_winner = prediction_data.get('predicted_winner')
            if not predicted_winner:
                return signals
            
            # Get odds for predicted winner
            winner_odds = None
            if odds_data and 'match_winner' in odds_data:
                winner_odds = odds_data['match_winner'].get(predicted_winner)
            
            if not winner_odds:
                winner_odds = 2.0  # Default odds if not available
            
            # Calculate ML edge
            model_prob = prediction_data.get('probabilities', {}).get(predicted_winner, 0.5)
            implied_prob = 1.0 / winner_odds
            edge = (model_prob - implied_prob) * self.config.ml_edge_multiplier
            ev = (model_prob * winner_odds) - 1
            
            if ev < self.config.min_ev:
                return signals
            
            # Calculate stake
            kelly = self.kelly_calc.calculate_kelly_fraction(
                probability=model_prob,
                odds=winner_odds,
                multiplier=self.config.kelly_multiplier
            )
            
            stake = min(kelly, self.config.max_stake)
            priority = self._determine_priority(ev)
            
            signal = BettingSignal(
                match_id=match_id,
                side=SignalSide(predicted_winner.lower()),
                stake=stake,
                ev=ev,
                confidence=confidence,
                odds=winner_odds,
                probability=model_prob,
                kelly_fraction=kelly,
                source="ml_prediction",
                strategy=SignalStrategy.ML_PREDICTION.value,
                priority=priority,
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.signal_expiry_minutes),
                reasons=[
                    f"ML prediction: {predicted_winner}",
                    f"Model confidence: {confidence:.2%}",
                    f"Calculated edge: {edge:.2%}",
                    f"Expected value: {ev:.2%}"
                ],
                metadata={
                    "model_version": prediction_data.get('model_version'),
                    "feature_count": prediction_data.get('feature_count'),
                    "training_accuracy": prediction_data.get('training_accuracy'),
                    "edge_multiplier": self.config.ml_edge_multiplier
                }
            )
            
            signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
        
        return signals
    
    async def _generate_pattern_signals(
        self, 
        match_data: Dict[str, Any], 
        odds_data: Dict[str, Any] = None
    ) -> List[BettingSignal]:
        """Generate pattern-based signals"""
        signals = []
        
        try:
            # This would analyze historical patterns, team performance, etc.
            # For now, return empty list - to be implemented with historical data
            pass
            
        except Exception as e:
            logger.error(f"Error generating pattern signals: {e}")
        
        return signals
    
    def _calculate_arbitrage(self, bookmaker_odds: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Calculate arbitrage opportunity"""
        try:
            if len(bookmaker_odds) < 2:
                return None
            
            # Find best odds for each outcome
            outcomes = set()
            for bookmaker, odds_dict in bookmaker_odds.items():
                outcomes.update(odds_dict.keys())
            
            best_odds = {}
            for outcome in outcomes:
                best_odd = 0
                best_bookmaker = None
                
                for bookmaker, odds_dict in bookmaker_odds.items():
                    if outcome in odds_dict and odds_dict[outcome] > best_odd:
                        best_odd = odds_dict[outcome]
                        best_bookmaker = bookmaker
                
                if best_bookmaker:
                    best_odds[outcome] = {
                        'odds': best_odd,
                        'bookmaker': best_bookmaker
                    }
            
            if len(best_odds) < 2:
                return None
            
            # Calculate if arbitrage exists
            total_inverse_odds = sum(1.0 / info['odds'] for info in best_odds.values())
            
            if total_inverse_odds >= 1.0:
                return None  # No arbitrage opportunity
            
            profit_margin = 1.0 - total_inverse_odds
            total_stake = 1000.0  # Base stake for calculation
            
            bets = {}
            for outcome, info in best_odds.items():
                stake = (total_stake / info['odds']) / total_inverse_odds
                bets[outcome] = {
                    'stake': stake / total_stake,  # As fraction of bankroll
                    'odds': info['odds'],
                    'bookmaker': info['bookmaker']
                }
            
            return {
                'profit': profit_margin,
                'total_return': 1.0 + profit_margin,
                'total_stake': 1.0,
                'bets': bets,
                'group_id': str(uuid.uuid4())[:8]
            }
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage: {e}")
            return None
    
    def _determine_priority(self, ev: float) -> SignalPriority:
        """Determine signal priority based on EV"""
        if ev >= self.config.critical_priority_ev:
            return SignalPriority.CRITICAL
        elif ev >= self.config.high_priority_ev:
            return SignalPriority.HIGH
        else:
            return SignalPriority.MEDIUM
    
    async def _validate_signals(self, signals: List[BettingSignal]) -> List[BettingSignal]:
        """Validate signals before publishing"""
        valid_signals = []
        
        for signal in signals:
            try:
                # Basic validation
                if signal.stake <= 0 or signal.ev <= 0:
                    continue
                
                if signal.confidence < self.config.min_confidence:
                    continue
                
                if signal.odds <= 1.0:
                    continue
                
                # Check if signal already exists for this match
                existing_key = f"{signal.match_id}_{signal.side.value}_{signal.strategy}"
                if existing_key in self.active_signals:
                    continue
                
                valid_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error validating signal: {e}")
                continue
        
        return valid_signals
    
    async def _apply_risk_management(self, signals: List[BettingSignal]) -> List[BettingSignal]:
        """Apply risk management rules"""
        if not signals:
            return signals
        
        # Sort by priority and EV
        signals.sort(key=lambda s: (s.priority.value, s.ev), reverse=True)
        
        final_signals = []
        projected_exposure = self.total_exposure
        
        for signal in signals:
            # Check exposure limits
            signal_exposure = signal.stake
            if projected_exposure + signal_exposure > self.config.max_exposure:
                continue
            
            # Check concurrent signals limit
            if len(final_signals) >= self.config.max_concurrent_signals:
                break
            
            final_signals.append(signal)
            projected_exposure += signal_exposure
        
        return final_signals
    
    async def _publish_signal(self, signal: BettingSignal) -> bool:
        """Publish signal to Redis"""
        if not self.redis_client:
            logger.debug("Redis not available, signal not published")
            return False
        
        try:
            signal_data = {
                "signal_id": str(uuid.uuid4()),
                "publisher_id": self.publisher_id,
                "timestamp": datetime.utcnow().isoformat(),
                "match_id": signal.match_id,
                "side": signal.side.value,
                "stake": signal.stake,
                "ev": signal.ev,
                "confidence": signal.confidence,
                "odds": signal.odds,
                "probability": signal.probability,
                "kelly_fraction": signal.kelly_fraction,
                "source": signal.source,
                "strategy": signal.strategy,
                "priority": signal.priority.value,
                "expires_at": signal.expires_at.isoformat() if signal.expires_at else None,
                "reasons": signal.reasons,
                "metadata": signal.metadata or {}
            }
            
            # Multi-channel publishing
            await asyncio.gather(
                # Queue for processing
                self.redis_client.lpush(self.signal_queue, json.dumps(signal_data)),
                # History storage
                self.redis_client.setex(
                    f"{self.signal_history}:{signal_data['signal_id']}", 
                    86400, 
                    json.dumps(signal_data)
                ),
                # Live pub/sub
                self.redis_client.publish(f"{self.signal_queue}:live", json.dumps(signal_data)),
                # Active signals tracking
                self.redis_client.setex(
                    f"{self.signal_active}:{signal.match_id}_{signal.side.value}",
                    self.config.signal_expiry_minutes * 60,
                    json.dumps(signal_data)
                )
            )
            
            self.metrics.signals_published_total.inc()
            logger.info(f"Signal published: {signal.match_id} - {signal.side.value} - EV: {signal.ev:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")
            self.metrics.signal_publish_errors_total.inc()
            return False
    
    async def _track_signal(self, signal: BettingSignal):
        """Track signal internally"""
        key = f"{signal.match_id}_{signal.side.value}_{signal.strategy}"
        self.active_signals[key] = signal
        self.daily_signal_count += 1
        self.total_exposure += signal.stake
        
        # Set cooldown
        await self.cooldown_mgr.set_match_cooldown(
            signal.match_id, 
            self.config.signal_expiry_minutes
        )
    
    async def _load_active_signals(self):
        """Load active signals from Redis on startup"""
        try:
            if not self.redis_client:
                return
            
            pattern = f"{self.signal_active}:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                signal_data = await self.redis_client.get(key)
                if signal_data:
                    data = json.loads(signal_data)
                    # Reconstruct tracking info
                    tracking_key = f"{data['match_id']}_{data['side']}_{data['strategy']}"
                    self.total_exposure += data.get('stake', 0)
                    
        except Exception as e:
            logger.error(f"Error loading active signals: {e}")
    
    async def _cleanup_expired_signals(self):
        """Background task to cleanup expired signals"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, signal in self.active_signals.items():
                    if signal.expires_at and current_time > signal.expires_at:
                        expired_keys.append(key)
                        self.total_exposure -= signal.stake
                
                for key in expired_keys:
                    del self.active_signals[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired signals")
                    
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _daily_reset_task(self):
        """Reset daily counters"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                current_date = datetime.utcnow().date()
                if current_date > self.last_reset:
                    self.daily_signal_count = 0
                    self.processed_matches.clear()
                    self.last_reset = current_date
                    logger.info("Daily signal counters reset")
                    
            except Exception as e:
                logger.error(f"Error in daily reset task: {e}")
    
    async def _check_daily_limits(self) -> bool:
        """Check if daily limits are exceeded"""
        if self.daily_signal_count >= self.config.max_daily_signals:
            logger.warning("Daily signal limit reached")
            return False
        
        if self.total_exposure >= self.config.max_exposure:
            logger.warning("Maximum exposure reached")
            return False
        
        return True
    
    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get currently active signals"""
        return [asdict(signal) for signal in self.active_signals.values()]
    
    async def get_signal_metrics(self) -> Dict[str, Any]:
        """Get signal generation metrics"""
        return {
            "daily_signal_count": self.daily_signal_count,
            "active_signals": len(self.active_signals),
            "total_exposure": self.total_exposure,
            "max_exposure": self.config.max_exposure,
            "exposure_utilization": self.total_exposure / self.config.max_exposure,
            "processed_matches": len(self.processed_matches)
        }
