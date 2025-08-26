import asyncio
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from dataclasses import dataclass, asdict

from models.signal import BettingSignal, SignalSide, SignalPriority

# Note: numpy used only for potential future calcs; not strictly required in current functions
from utils.kelly import calculate_kelly_fraction

# Redis imports with fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    min_ev: float = 0.05  # Minimum EV threshold (5%)
    min_confidence: float = 0.65  # Minimum confidence (65%)
    max_stake: float = 0.10  # Maximum stake (10% of bankroll)
    kelly_multiplier: float = 0.25  # Kelly fraction multiplier (quarter Kelly)

    # Signal filters
    enable_value_betting: bool = True
    enable_arbitrage: bool = True
    enable_ml_signals: bool = True
    enable_pattern_signals: bool = True

    # Risk management
    max_concurrent_signals: int = 5
    max_exposure: float = 0.30  # Maximum total exposure (30%)
    signal_expiry_minutes: int = 5

    # Priority thresholds
    high_priority_ev: float = 0.15
    critical_priority_ev: float = 0.25


class SignalGenerator:
    """Professional Signal Generator with Redis Publishing"""

    def __init__(self, config: SignalConfig = None, redis_url: str = "redis://localhost:6379"):
        self.config = config or SignalConfig()
        self.active_signals: Dict[str, BettingSignal] = {}
        self._lock = asyncio.Lock()
        
        # Redis setup
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.signal_queue = "cs2_betting_signals"
        self.signal_history = "cs2_signal_history"
        self.publisher_id = str(uuid.uuid4())[:8]
        
    async def initialize(self):
        """Initialize Redis connection"""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info(f"SignalGenerator connected to Redis: {self.redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Signals will not be published.")
                self.redis_client = None
        else:
            logger.warning("Redis not available. Signals will not be published.")
            
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def publish_signal(self, signal: BettingSignal) -> bool:
        """Publish signal to Redis queue with persistence"""
        if not self.redis_client:
            logger.debug("Redis not available, signal not published")
            return False
            
        try:
            # Create signal payload
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
            
            # Publish to queue (LPUSH for FIFO processing)
            await self.redis_client.lpush(self.signal_queue, json.dumps(signal_data))
            
            # Store in history with expiration (24 hours)
            history_key = f"{self.signal_history}:{signal_data['signal_id']}"
            await self.redis_client.setex(history_key, 86400, json.dumps(signal_data))
            
            # Publish to pub/sub for real-time consumers
            await self.redis_client.publish(f"{self.signal_queue}_live", json.dumps(signal_data))
            
            logger.info(f"Signal published: {signal.match_id} - {signal.side.value} - EV: {signal.ev:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")
            return False

    async def generate_value_signal(
        self,
        match_id: str,
        our_prob: float,
        market_odds: float,
        side: SignalSide,
        source: str = "value_betting",
    ) -> Optional[BettingSignal]:
        """สร้างสัญญาณจาก Value Betting"""
        try:
            # Calculate EV
            ev = (our_prob * market_odds) - 1

            if ev < self.config.min_ev:
                return None

            # Calculate confidence based on edge size
            confidence = min(0.95, 0.5 + (ev * 2))

            if confidence < self.config.min_confidence:
                return None

            # Calculate Kelly fraction
            kelly = calculate_kelly_fraction(
                probability=our_prob, odds=market_odds, multiplier=self.config.kelly_multiplier
            )

            # Determine stake
            stake = min(kelly, self.config.max_stake)

            # Set priority
            priority = self._determine_priority(ev)

            # Create signal
            signal = BettingSignal(
                match_id=match_id,
                side=side,
                stake=stake,
                ev=ev,
                confidence=confidence,
                odds=market_odds,
                probability=our_prob,
                kelly_fraction=kelly,
                source=source,
                strategy="value_betting",
                priority=priority,
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.signal_expiry_minutes),
                reasons=[
                    f"Value edge detected: {ev:.2%}",
                    f"Our probability: {our_prob:.2%}",
                    f"Market odds: {market_odds:.2f}",
                    f"Kelly stake: {kelly:.2%}",
                ],
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating value signal: {e}")
            return None

    async def generate_arbitrage_signal(
        self, match_id: str, odds_dict: Dict[str, float], source: str = "arbitrage"
    ) -> List[BettingSignal]:
        """สร้างสัญญาณ Arbitrage จากราคาต่างเว็บ"""
        signals: List[BettingSignal] = []

        try:
            # Check for arbitrage opportunity
            arb_info = self._check_arbitrage(odds_dict)

            if not arb_info or arb_info["profit"] < self.config.min_ev:
                return signals

            # Generate signals for each leg
            for side, stake_pct in arb_info["stakes"].items():
                signal = BettingSignal(
                    match_id=match_id,
                    side=SignalSide(side),
                    stake=stake_pct,
                    ev=arb_info["profit"],
                    confidence=0.99,  # High confidence for arbitrage
                    odds=odds_dict[side],
                    source=source,
                    strategy="arbitrage",
                    priority=SignalPriority.CRITICAL,
                    expires_at=datetime.utcnow() + timedelta(minutes=2),
                    reasons=[
                        f"Arbitrage opportunity: {arb_info['profit']:.2%}",
                        f"Total stake required: {sum(arb_info['stakes'].values()):.2%}",
                        "Guaranteed profit regardless of outcome",
                    ],
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating arbitrage signal: {e}")

        return signals

    async def generate_ml_signal(
        self, match_id: str, predictions: Dict[str, Any], source: str = "ml_model"
    ) -> Optional[BettingSignal]:
        """สร้างสัญญาณจาก ML Model"""
        try:
            # Extract predictions
            predicted_prob = predictions.get("probability", 0)
            confidence = predictions.get("confidence", 0)
            suggested_side = predictions.get("side")
            market_odds = predictions.get("market_odds", 0)

            # Validate
            if confidence < self.config.min_confidence:
                return None

            # Calculate EV
            ev = (predicted_prob * market_odds) - 1

            if ev < self.config.min_ev:
                return None

            # Calculate stake
            kelly = calculate_kelly_fraction(
                probability=predicted_prob, odds=market_odds, multiplier=self.config.kelly_multiplier
            )
            stake = min(kelly, self.config.max_stake)

            # Create signal
            signal = BettingSignal(
                match_id=match_id,
                side=SignalSide(suggested_side),
                stake=stake,
                ev=ev,
                confidence=confidence,
                odds=market_odds,
                probability=predicted_prob,
                kelly_fraction=kelly,
                source=source,
                strategy="ml_prediction",
                priority=self._determine_priority(ev),
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.signal_expiry_minutes),
                metadata={
                    "model_name": predictions.get("model_name"),
                    "model_version": predictions.get("model_version"),
                    "features_used": predictions.get("features_used", []),
                },
                reasons=predictions.get("reasons", []),
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return None

    async def generate_pattern_signal(
        self, match_id: str, pattern_data: Dict[str, Any], source: str = "pattern_recognition"
    ) -> Optional[BettingSignal]:
        """สร้างสัญญาณจาก Pattern Recognition"""
        try:
            pattern_type = pattern_data.get("pattern_type")
            pattern_confidence = pattern_data.get("confidence", 0)
            expected_outcome = pattern_data.get("expected_outcome")
            historical_accuracy = pattern_data.get("historical_accuracy", 0)

            # Calculate overall confidence
            confidence = pattern_confidence * historical_accuracy

            if confidence < self.config.min_confidence:
                return None

            # Get market odds and calculate EV
            market_odds = pattern_data.get("current_odds", 0)
            implied_prob = 1 / market_odds if market_odds > 0 else 0
            our_prob = pattern_data.get("expected_probability", implied_prob * 1.1)

            ev = (our_prob * market_odds) - 1

            if ev < self.config.min_ev:
                return None

            # Calculate stake
            stake = min(self.config.max_stake * confidence, self.config.max_stake)

            # Create signal
            signal = BettingSignal(
                match_id=match_id,
                side=SignalSide(expected_outcome),
                stake=stake,
                ev=ev,
                confidence=confidence,
                odds=market_odds,
                probability=our_prob,
                source=source,
                strategy=f"pattern_{pattern_type}",
                priority=self._determine_priority(ev),
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.signal_expiry_minutes),
                metadata={
                    "pattern_type": pattern_type,
                    "pattern_strength": pattern_data.get("strength"),
                    "historical_occurrences": pattern_data.get("occurrences", 0),
                },
                reasons=[
                    f"Pattern detected: {pattern_type}",
                    f"Historical accuracy: {historical_accuracy:.2%}",
                    f"Pattern confidence: {pattern_confidence:.2%}",
                ],
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating pattern signal: {e}")
            return None

    def _determine_priority(self, ev: float) -> SignalPriority:
        """กำหนด priority ตาม EV"""
        if ev >= self.config.critical_priority_ev:
            return SignalPriority.CRITICAL
        elif ev >= self.config.high_priority_ev:
            return SignalPriority.HIGH
        elif ev >= 0.10:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW

    def _check_arbitrage(self, odds_dict: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """ตรวจสอบโอกาส Arbitrage"""
        try:
            # Calculate implied probabilities
            total_implied = sum(1 / odds for odds in odds_dict.values() if odds > 0)

            if total_implied >= 1:
                return None  # No arbitrage

            # Calculate profit and stakes
            profit = (1 - total_implied) / total_implied
            stakes: Dict[str, float] = {}

            for side, odds in odds_dict.items():
                if odds <= 0:
                    continue
                stake = (1 / odds) / total_implied
                stakes[side] = stake

            return {"profit": profit, "stakes": stakes, "total_implied": total_implied}

        except Exception as e:
            logger.error(f"Error checking arbitrage: {e}")
            return None

    async def validate_signal(self, signal: BettingSignal) -> bool:
        """ตรวจสอบความถูกต้องของสัญญาณ"""
        async with self._lock:
            # Check if match already has signal
            if signal.match_id in self.active_signals:
                existing = self.active_signals[signal.match_id]
                # Allow if better EV
                if signal.ev <= existing.ev:
                    return False

            # Check total exposure
            total_exposure = sum(s.stake for s in self.active_signals.values())

            if total_exposure + signal.stake > self.config.max_exposure:
                logger.warning("Signal rejected: Would exceed max exposure")
                return False

            # Check concurrent signals limit
            if len(self.active_signals) >= self.config.max_concurrent_signals:
                # Remove expired signals
                now = datetime.utcnow()
                expired = [sid for sid, s in self.active_signals.items() if s.expires_at and s.expires_at < now]
                for sid in expired:
                    del self.active_signals[sid]

                # Check again
                if len(self.active_signals) >= self.config.max_concurrent_signals:
                    logger.warning("Signal rejected: Max concurrent signals reached")
                    return False

            return True

    async def add_active_signal(self, signal: BettingSignal):
        """Add signal to active list and publish to Redis"""
        async with self._lock:
            self.active_signals[signal.match_id] = signal
            
        # Publish signal
        await self.publish_signal(signal)
        
    async def process_and_publish_signal(self, signal: BettingSignal) -> bool:
        """Validate, add, and publish signal in one operation"""
        # Validate signal
        if not await self.validate_signal(signal):
            logger.info(f"Signal validation failed: {signal.match_id}")
            return False
            
        # Add to active signals and publish
        await self.add_active_signal(signal)
        
        logger.info(f"Signal processed and published: {signal.match_id} - {signal.strategy} - EV: {signal.ev:.2%}")
        return True
        
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        if not self.redis_client:
            return {"queue_length": 0, "redis_connected": False}
            
        try:
            queue_length = await self.redis_client.llen(self.signal_queue)
            return {
                "queue_length": queue_length,
                "redis_connected": True,
                "active_signals": len(self.active_signals),
                "publisher_id": self.publisher_id
            }
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {"queue_length": 0, "redis_connected": False, "error": str(e)}
            
    async def clear_expired_signals(self) -> int:
        """Remove expired signals from active list"""
        async with self._lock:
            now = datetime.utcnow()
            expired_ids = [
                signal_id for signal_id, signal in self.active_signals.items()
                if signal.expires_at and signal.expires_at < now
            ]
            
            for signal_id in expired_ids:
                del self.active_signals[signal_id]
                
            if expired_ids:
                logger.info(f"Cleared {len(expired_ids)} expired signals")
                
            return len(expired_ids)
