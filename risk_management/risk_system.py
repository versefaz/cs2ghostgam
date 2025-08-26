import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

from .kelly_calculator import KellyCalculator, KellyParameters, BettingLimits
from .cooldown_manager import CooldownConfig, CooldownManager, CooldownLevel
from .position_manager import PositionManager, Position


@dataclass
class RiskSignal:
    """Risk-adjusted betting signal"""
    original_signal: Dict
    kelly_stake: float
    risk_score: float
    adjustments: List[str]
    can_execute: bool
    rejection_reason: Optional[str] = None


class RiskManagementSystem:
    """Complete risk management system"""

    def __init__(self, bankroll: float, limits: BettingLimits, kelly_fraction: float = 0.25):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.limits = limits
        self.kelly_calculator = KellyCalculator(limits)
        self.cooldown_config = CooldownConfig()
        self.cooldown_manager = CooldownManager(self.cooldown_config)
        self.position_manager = PositionManager(limits, self.cooldown_manager)
        self.kelly_fraction = kelly_fraction
        self.logger = logging.getLogger(__name__)

    async def process_signal(self, signal: Dict) -> RiskSignal:
        """Process betting signal through risk management"""
        try:
            # 1. Extract signal parameters
            probability = max(0.0, min(1.0, signal.get("confidence", 0) / 100))
            odds = float(signal.get("odds", 1.0))
            match_id = signal["match_id"]
            team_id = signal.get("team", "")
            market = signal.get("market_type", "match_winner")

            # 2. Calculate Kelly stake
            kelly_params = KellyParameters(
                probability=probability,
                odds=odds,
                bankroll=self.bankroll,
                kelly_fraction=self.kelly_fraction,
                edge_confidence=self._calculate_edge_confidence(signal),
            )

            kelly_stake = self.kelly_calculator.calculate_kelly(kelly_params)

            # 3. Check if we can place the bet
            can_place, reason = await self.position_manager.can_place_bet(
                match_id=match_id, team_id=team_id, market=market, stake=kelly_stake
            )

            # 4. Calculate risk score
            risk_score = self._calculate_risk_score(signal, kelly_stake)

            # 5. Build risk-adjusted signal
            adjustments: List[str] = []

            # Apply volatility adjustment
            if signal.get("volatility", 0) > 0.3:
                kelly_stake *= 0.7
                adjustments.append("High volatility reduction: 30%")

            # Apply correlation adjustment
            correlation_factor = await self._check_correlation(signal)
            if correlation_factor < 1.0:
                kelly_stake *= correlation_factor
                adjustments.append(f"Correlation adjustment: {correlation_factor:.2f}")

            # Final stake validation
            if kelly_stake < self.limits.min_bet_size:
                can_place = False
                reason = "Stake below minimum after adjustments"

            return RiskSignal(
                original_signal=signal,
                kelly_stake=round(kelly_stake, 2),
                risk_score=risk_score,
                adjustments=adjustments,
                can_execute=can_place and risk_score < 0.7,
                rejection_reason=reason if not can_place else None,
            )

        except Exception as e:
            self.logger.error(f"Risk processing error: {e}")
            return RiskSignal(
                original_signal=signal,
                kelly_stake=0,
                risk_score=1.0,
                adjustments=["Error in processing"],
                can_execute=False,
                rejection_reason=str(e),
            )

    def _calculate_edge_confidence(self, signal: Dict) -> float:
        """Calculate confidence in our edge"""
        confidence_factors = {
            "model_confidence": signal.get("model_confidence", 50) / 100,
            "data_quality": signal.get("data_quality", 0.5),
            "sample_size": min(signal.get("sample_size", 0) / 1000, 1.0),
            "recency": 1.0 - (signal.get("data_age_hours", 0) / 168),  # Week old = 0
        }

        # Weighted average
        weights = {"model_confidence": 0.4, "data_quality": 0.3, "sample_size": 0.2, "recency": 0.1}

        edge_confidence = sum(confidence_factors[k] * weights[k] for k in confidence_factors)

        return max(0.1, min(1.0, edge_confidence))

    def _calculate_risk_score(self, signal: Dict, stake: float) -> float:
        """Calculate overall risk score (0-1, lower is better)"""
        risk_factors: List[float] = []

        # 1. Stake as percentage of bankroll
        stake_pct = 0.0 if self.bankroll <= 0 else (stake / self.bankroll)
        risk_factors.append(min(stake_pct * 10, 1.0))  # 10% = max risk

        # 2. Odds risk (higher odds = higher risk)
        odds = float(signal.get("odds", 1.0))
        odds_risk = min(max((odds - 1) / 9, 0.0), 1.0)  # Odds of 10 = max risk
        risk_factors.append(odds_risk)

        # 3. Confidence inverse
        confidence = max(0.0, min(1.0, signal.get("confidence", 50) / 100))
        risk_factors.append(1 - confidence)

        # 4. Market volatility
        volatility = max(0.0, min(1.0, signal.get("volatility", 0.5)))
        risk_factors.append(volatility)

        # 5. Time to event (closer = riskier for live)
        if signal.get("is_live", False):
            time_remaining = max(0, int(signal.get("time_remaining_min", 90)))
            time_risk = 1 - (time_remaining / 90)
            risk_factors.append(time_risk)

        # Calculate average
        risk_score = sum(risk_factors) / max(1, len(risk_factors))

        return min(1.0, max(0.0, risk_score))

    async def _check_correlation(self, signal: Dict) -> float:
        """
        Check correlation with existing positions
        Returns adjustment factor (0-1)
        """
        correlation_penalty = 0.0

        # Check team exposure
        team = signal.get("team", "")
        if team in self.position_manager.exposure.by_team:
            team_exposure = self.position_manager.exposure.by_team[team]
            if self.bankroll > 0 and team_exposure > self.bankroll * 0.1:  # >10% on same team
                correlation_penalty += 0.3

        # Check match exposure
        match_id = signal["match_id"]
        if match_id in self.position_manager.exposure.by_match:
            correlation_penalty += 0.5  # Heavy penalty for same match

        # Check market concentration
        market = signal.get("market_type", "match_winner")
        market_exposure = self.position_manager.exposure.by_market.get(market, 0.0)
        if self.bankroll > 0 and market_exposure > self.bankroll * 0.3:  # >30% in same market
            correlation_penalty += 0.2

        # Convert penalty to adjustment factor
        adjustment_factor = max(0.2, 1.0 - correlation_penalty)

        return adjustment_factor

    async def execute_signal(self, risk_signal: RiskSignal) -> bool:
        """Execute risk-approved signal"""
        if not risk_signal.can_execute:
            self.logger.info(f"Signal rejected: {risk_signal.rejection_reason}")
            return False

        position = Position(
            id=f"pos_{risk_signal.original_signal['id']}_{datetime.now().timestamp()}",
            match_id=risk_signal.original_signal["match_id"],
            team_id=risk_signal.original_signal.get("team", ""),
            market=risk_signal.original_signal.get("market_type", "match_winner"),
            stake=risk_signal.kelly_stake,
            odds=float(risk_signal.original_signal.get("odds", 1.0)),
            placed_at=datetime.now(),
        )

        success = await self.position_manager.place_bet(position)

        if success:
            self.logger.info(f"Position placed: {position.id}, stake: {position.stake}")
            # Update bankroll (in real system, after bet settlement)
            self.bankroll -= position.stake
        else:
            self.logger.warning(f"Failed to place position: {position.id}")

        return success
