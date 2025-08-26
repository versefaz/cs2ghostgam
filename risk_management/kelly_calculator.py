from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
from decimal import Decimal
from enum import Enum


class KellyMode(Enum):
    FULL = 1.0
    HALF = 0.5
    QUARTER = 0.25
    CONSERVATIVE = 0.2
    AGGRESSIVE = 0.75
    CUSTOM = None


@dataclass
class KellyParameters:
    """Parameters for Kelly calculation"""
    probability: float  # Win probability (0-1)
    odds: float  # Decimal odds
    bankroll: float  # Current bankroll
    kelly_fraction: float = 0.25  # Fractional Kelly (default 25%)
    edge_confidence: float = 1.0  # Confidence in edge (0-1)


@dataclass
class BettingLimits:
    """Betting constraints"""
    max_bet_size: float  # Maximum single bet
    max_exposure: float  # Maximum total exposure (absolute amount)
    max_bets_per_day: int  # Daily bet limit
    max_percent_bankroll: float = 0.05  # Max % per bet
    min_bet_size: float = 10  # Minimum bet size
    max_concurrent_bets: int = 10  # Max open positions


class KellyCalculator:
    """Advanced Kelly Criterion calculator with safety measures"""

    def __init__(self, limits: BettingLimits):
        self.limits = limits
        self.volatility_adjustments: Dict[str, float] = {}

    def calculate_kelly(self, params: KellyParameters) -> float:
        """
        Calculate Kelly stake with multiple safety checks

        Formula: f = (p*b - q) / b
        where:
            f = fraction of bankroll to bet
            p = probability of winning
            b = net odds (decimal odds - 1)
            q = probability of losing (1 - p)
        """
        p = params.probability
        b = params.odds - 1  # Net odds
        q = 1 - p

        if b <= 0 or p <= 0 or p >= 1:
            return 0.0

        # Basic Kelly formula
        kelly_fraction = (p * b - q) / b

        # Apply edge confidence adjustment
        kelly_fraction *= max(0.0, min(1.0, params.edge_confidence))

        # Apply fractional Kelly
        kelly_fraction *= max(0.0, params.kelly_fraction)

        # Calculate actual bet size
        bet_size = kelly_fraction * params.bankroll

        # Apply multiple constraints
        bet_size = self._apply_constraints(bet_size, params)

        return bet_size

    def _apply_constraints(self, bet_size: float, params: KellyParameters) -> float:
        """Apply all betting constraints"""

        # 1. Maximum bet size limit
        bet_size = min(bet_size, self.limits.max_bet_size)

        # 2. Maximum percentage of bankroll
        max_from_bankroll = params.bankroll * self.limits.max_percent_bankroll
        bet_size = min(bet_size, max_from_bankroll)

        # 3. Minimum bet size
        if bet_size < self.limits.min_bet_size:
            return 0  # Don't bet if below minimum

        # 4. Round to sensible amount
        bet_size = round(bet_size, 2)

        return bet_size

    def calculate_multi_kelly(self, opportunities: List[KellyParameters]) -> Dict[int, float]:
        """
        Calculate Kelly for multiple simultaneous opportunities
        Using correlation adjustments (greedy by EV)
        """
        results: Dict[int, float] = {}
        total_allocation = 0.0

        # Sort by expected value
        sorted_opps = sorted(
            enumerate(opportunities), key=lambda x: self._calculate_ev(x[1]), reverse=True
        )

        for idx, params in sorted_opps:
            # Check if we can allocate more
            remaining_exposure = self.limits.max_exposure - total_allocation

            if remaining_exposure <= 0:
                results[idx] = 0.0
                continue

            # Calculate Kelly for this opportunity
            kelly_bet = self.calculate_kelly(params)

            # Apply remaining exposure limit
            kelly_bet = float(min(kelly_bet, remaining_exposure))

            results[idx] = kelly_bet
            total_allocation += kelly_bet

        return results

    def _calculate_ev(self, params: KellyParameters) -> float:
        """Calculate expected value"""
        return (params.probability * params.odds) - 1
