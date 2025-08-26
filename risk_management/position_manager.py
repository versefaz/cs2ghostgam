from typing import Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from collections import defaultdict

from .kelly_calculator import BettingLimits
from .cooldown_manager import CooldownManager, CooldownLevel


@dataclass
class Position:
    """Active betting position"""
    id: str
    match_id: str
    team_id: str
    market: str
    stake: float
    odds: float
    placed_at: datetime
    status: str = "pending"
    pnl: float = 0.0


@dataclass
class ExposureState:
    """Current exposure state"""
    total_exposure: float = 0.0
    positions_count: int = 0
    daily_bet_count: int = 0
    by_match: Dict[str, float] = field(default_factory=dict)
    by_team: Dict[str, float] = field(default_factory=dict)
    by_market: Dict[str, float] = field(default_factory=dict)
    by_league: Dict[str, float] = field(default_factory=dict)


class PositionManager:
    """Manage positions and exposure limits"""

    def __init__(self, limits: BettingLimits, cooldown_manager: CooldownManager):
        self.limits = limits
        self.cooldown_manager = cooldown_manager
        self.positions: Dict[str, Position] = {}
        self.exposure = ExposureState()
        self.daily_bets = defaultdict(int)
        self._lock = asyncio.Lock()

    async def can_place_bet(
        self, match_id: str, team_id: str, market: str, stake: float
    ) -> Tuple[bool, str]:
        """Check if bet can be placed within all constraints"""
        async with self._lock:
            # 1. Check cooldowns
            cooldown_checks = [
                (CooldownLevel.MATCH, match_id),
                (CooldownLevel.TEAM, team_id),
                (CooldownLevel.MARKET, market),
            ]

            for level, entity_id in cooldown_checks:
                available, remaining = self.cooldown_manager.is_available(level, entity_id)
                if not available:
                    remaining_s = f"{remaining}" if remaining is not None else "unknown"
                    return False, f"Cooldown active for {level.value}: {remaining_s}"

            # 2. Check exposure limits (absolute amount)
            if self.exposure.total_exposure + stake > self.limits.max_exposure:
                return False, f"Would exceed max exposure: {self.limits.max_exposure}"

            # 3. Check daily bet limit
            today = datetime.now().date()
            if self.daily_bets[today] >= self.limits.max_bets_per_day:
                return False, f"Daily bet limit reached: {self.limits.max_bets_per_day}"

            # 4. Check concurrent positions
            if self.exposure.positions_count >= self.limits.max_concurrent_bets:
                return False, f"Max concurrent bets reached: {self.limits.max_concurrent_bets}"

            # 5. Check single bet size
            if stake > self.limits.max_bet_size:
                return False, f"Bet size exceeds maximum: {self.limits.max_bet_size}"

            # 6. Check minimum bet size
            if stake < self.limits.min_bet_size:
                return False, f"Bet size below minimum: {self.limits.min_bet_size}"

            return True, "OK"

    async def place_bet(self, position: Position) -> bool:
        """Place bet if allowed by risk management"""
        can_place, reason = await self.can_place_bet(
            position.match_id, position.team_id, position.market, position.stake
        )

        if not can_place:
            print(f"Bet rejected: {reason}")
            return False

        async with self._lock:
            # Add position
            self.positions[position.id] = position

            # Update exposure
            self.exposure.total_exposure += position.stake
            self.exposure.positions_count += 1
            self.exposure.by_match[position.match_id] = (
                self.exposure.by_match.get(position.match_id, 0.0) + position.stake
            )
            self.exposure.by_team[position.team_id] = (
                self.exposure.by_team.get(position.team_id, 0.0) + position.stake
            )
            self.exposure.by_market[position.market] = (
                self.exposure.by_market.get(position.market, 0.0) + position.stake
            )

            # Update daily count
            today = datetime.now().date()
            self.daily_bets[today] += 1

            # Add cooldowns
            self.cooldown_manager.add_cooldown(CooldownLevel.MATCH, position.match_id)

            return True

    async def close_position(self, position_id: str, result: str, pnl: float):
        """Close position and update cooldowns"""
        async with self._lock:
            if position_id not in self.positions:
                return

            position = self.positions[position_id]
            position.status = "closed"
            position.pnl = pnl

            # Update exposure
            self.exposure.total_exposure -= position.stake
            self.exposure.positions_count -= 1
            self.exposure.by_match[position.match_id] -= position.stake
            self.exposure.by_team[position.team_id] -= position.stake
            self.exposure.by_market[position.market] -= position.stake

            # Update cooldowns based on result
            if result == "loss":
                # Longer cooldowns on losses
                self.cooldown_manager.add_cooldown(
                    CooldownLevel.TEAM, position.team_id, result="loss"
                )
                self.cooldown_manager.add_cooldown(
                    CooldownLevel.MARKET, position.market, result="loss"
                )

            # Remove closed position
            del self.positions[position_id]
