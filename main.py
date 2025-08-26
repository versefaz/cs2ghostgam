import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict

from risk_management.kelly_calculator import BettingLimits, KellyParameters
from risk_management.risk_system import RiskManagementSystem
from risk_management.monitoring import RiskMonitor
from risk_management.cooldown_manager import CooldownLevel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Complete risk management system demo"""

    # 1. Initialize system
    bankroll = 10000.0
    limits = BettingLimits(
        max_bet_size=500.0,
        max_exposure=2000.0,
        max_bets_per_day=20,
        max_percent_bankroll=0.05,
        min_bet_size=10.0,
        max_concurrent_bets=10,
    )

    risk_system = RiskManagementSystem(
        bankroll=bankroll, limits=limits, kelly_fraction=0.25
    )

    # 2. Start monitoring
    monitor = RiskMonitor(risk_system)
    monitor_task = asyncio.create_task(monitor.monitor_loop())

    # 3. Process multiple signals
    signals: List[Dict] = [
        {
            "id": "sig_123",
            "match_id": "match_456",
            "team": "NaVi",
            "market_type": "match_winner",
            "confidence": 65,  # 65% win probability
            "odds": 2.1,
            "volatility": 0.2,
            "model_confidence": 70,
            "data_quality": 0.8,
            "sample_size": 500,
            "data_age_hours": 24,
            "is_live": False,
        },
        {
            "id": "sig_124",
            "match_id": "match_457",
            "team": "FaZe",
            "market_type": "first_map",
            "confidence": 58,
            "odds": 1.85,
            "volatility": 0.35,
            "model_confidence": 60,
            "data_quality": 0.7,
            "sample_size": 300,
            "data_age_hours": 12,
            "is_live": False,
        },
        {
            "id": "sig_125",
            "match_id": "match_458",
            "team": "G2",
            "market_type": "over_2.5_maps",
            "confidence": 72,
            "odds": 1.95,
            "volatility": 0.15,
            "model_confidence": 75,
            "data_quality": 0.9,
            "sample_size": 1000,
            "data_age_hours": 6,
            "is_live": False,
        },
    ]

    executed_positions = []
    for signal in signals:
        print(f"\n{'='*60}")
        print(f"Processing Signal: {signal['id']}")
        print(f"{'='*60}")

        # Process through risk management
        risk_signal = await risk_system.process_signal(signal)

        # Display analysis
        print("\n\uD83D\uDCCA Signal Analysis:")
        print(f"  Match: {signal['match_id']}")
        print(f"  Team: {signal['team']}")
        print(f"  Market: {signal['market_type']}")
        print(f"  Confidence: {signal['confidence']}%")
        print(f"  Odds: {signal['odds']}")

        print("\n\uD83D\uDCB0 Kelly Calculation:")
        print(f"  Kelly Stake: ${risk_signal.kelly_stake}")
        print(f"  Risk Score: {risk_signal.risk_score:.2f}")
        print(f"  Can Execute: {risk_signal.can_execute}")

        if risk_signal.adjustments:
            print("\n\u26A0\uFE0F Adjustments Applied:")
            for adj in risk_signal.adjustments:
                print(f"  - {adj}")

        if risk_signal.rejection_reason:
            print(f"\n\u274C Rejection Reason: {risk_signal.rejection_reason}")

        # Try to execute
        if risk_signal.can_execute:
            success = await risk_system.execute_signal(risk_signal)
            if success:
                print(f"\u2705 Position Placed: ${risk_signal.kelly_stake}")
                executed_positions.append(risk_signal)
            else:
                print("\u26A0\uFE0F Position Failed")

        await asyncio.sleep(1)

    # 4. Show portfolio status
    print(f"\n{'='*60}")
    print("\uD83D\uDCC8 PORTFOLIO STATUS")
    print(f"{'='*60}")

    exposure = risk_system.position_manager.exposure
    print(f"Total Exposure: ${exposure.total_exposure:.2f}")
    print(f"Active Positions: {exposure.positions_count}")
    print(f"Available Bankroll: ${risk_system.bankroll:.2f}")
    print(
        f"Exposure % of Limit: {(exposure.total_exposure/limits.max_exposure)*100:.1f}%"
    )

    print(f"\n\uD83D\uDCCA Exposure Breakdown:")
    if exposure.by_match:
        print("By Match:")
        for match, amount in exposure.by_match.items():
            print(f"  {match}: ${amount:.2f}")

    if exposure.by_team:
        print("By Team:")
        for team, amount in exposure.by_team.items():
            print(f"  {team}: ${amount:.2f}")

    if exposure.by_market:
        print("By Market:")
        for market, amount in exposure.by_market.items():
            print(f"  {market}: ${amount:.2f}")

    # 5. Simulate bet settlement
    print(f"\n{'='*60}")
    print("\U0001F4B5 SIMULATING BET SETTLEMENT")
    print(f"{'='*60}")

    for position_id, position in list(risk_system.position_manager.positions.items()):
        import random

        won = random.random() < 0.55
        if won:
            pnl = position.stake * (position.odds - 1)
            result = "win"
            print(f"\u2705 WIN: {position_id[:10]}... PnL: +${pnl:.2f}")
        else:
            pnl = -position.stake
            result = "loss"
            print(f"\u274C LOSS: {position_id[:10]}... PnL: -${position.stake:.2f}")

        await risk_system.position_manager.close_position(position_id, result, pnl)
        risk_system.bankroll += pnl

    print(f"\n\uD83D\uDCB0 Updated Bankroll: ${risk_system.bankroll:.2f}")
    print(f"\ud83d\udcc9 Total PnL: ${risk_system.bankroll - bankroll:.2f}")

    # 6. Check cooldowns
    print(f"\n{'='*60}")
    print("\u23F0 COOLDOWN STATUS")
    print(f"{'='*60}")

    test_entities = [
        (CooldownLevel.TEAM, "NaVi"),
        (CooldownLevel.MATCH, "match_456"),
        (CooldownLevel.MARKET, "match_winner"),
    ]

    for level, entity in test_entities:
        available, remaining = risk_system.cooldown_manager.is_available(level, entity)
        if available:
            print(f"\u2705 {level.value}:{entity} - Available")
        else:
            print(f"\u23F0 {level.value}:{entity} - Cooldown: {remaining}")

    # 7. Check alerts
    if monitor.alerts:
        print(f"\n{'='*60}")
        print("\ud83d\udea8 RISK ALERTS")
        print(f"{'='*60}")
        for alert in monitor.alerts[-5:]:
            icon = "\U0001F534" if alert["level"] == "critical" else "\U0001F7E1"
            print(f"{icon} [{alert['level'].upper()}] {alert['message']}")
            if "action" in alert:
                print(f"   → Recommended: {alert['action']}")

    # Cleanup
    monitor_task.cancel()


async def advanced_example():
    """Advanced usage with multiple strategies and correlation-aware allocation"""

    # Different risk profiles
    profiles = {
        'conservative': {
            'kelly_fraction': 0.1,
            'max_bet_pct': 0.02,
            'max_exposure_pct': 0.1
        },
        'moderate': {
            'kelly_fraction': 0.25,
            'max_bet_pct': 0.05,
            'max_exposure_pct': 0.2
        },
        'aggressive': {
            'kelly_fraction': 0.5,
            'max_bet_pct': 0.1,
            'max_exposure_pct': 0.4
        }
    }

    bankroll = 50000.0
    profile = profiles['moderate']

    limits = BettingLimits(
        max_bet_size=bankroll * profile['max_bet_pct'],
        max_exposure=bankroll * profile['max_exposure_pct'],
        max_bets_per_day=30,
        max_percent_bankroll=profile['max_bet_pct'],
        min_bet_size=50.0,
        max_concurrent_bets=15
    )

    risk_system = RiskManagementSystem(
        bankroll=bankroll,
        limits=limits,
        kelly_fraction=profile['kelly_fraction']
    )

    # Process batch of signals with correlation management
    high_value_signals = [
        # Correlated signals (same match)
        {
            'id': 'hv_001',
            'match_id': 'finals_001',
            'team': 'TeamA',
            'market_type': 'match_winner',
            'confidence': 68,
            'odds': 2.2,
            'volatility': 0.25,
            'correlation_group': 'finals'
        },
        {
            'id': 'hv_002',
            'match_id': 'finals_001',
            'team': 'TeamA',
            'market_type': 'first_map',
            'confidence': 62,
            'odds': 1.9,
            'volatility': 0.3,
            'correlation_group': 'finals'
        },
        # Independent signal
        {
            'id': 'hv_003',
            'match_id': 'semi_001',
            'team': 'TeamB',
            'market_type': 'total_rounds_over',
            'confidence': 71,
            'odds': 1.85,
            'volatility': 0.2,
            'correlation_group': 'semis'
        }
    ]

    # Calculate multi-Kelly allocation
    kelly_params_list = []
    for sig in high_value_signals:
        params = KellyParameters(
            probability=sig['confidence']/100,
            odds=sig['odds'],
            bankroll=risk_system.bankroll,
            kelly_fraction=profile['kelly_fraction']
        )
        kelly_params_list.append(params)

    allocations = risk_system.kelly_calculator.calculate_multi_kelly(kelly_params_list)

    print("Multi-Kelly Allocation:")
    for idx, allocation in allocations.items():
        sig = high_value_signals[idx]
        print(f"  {sig['id']}: ${allocation:.2f} ({sig['market_type']})")


async def backtest_risk_management():
    """Backtest risk management on historical signals"""

    print(f"\n{'='*60}")
    print("\uD83D\uDCCA RISK MANAGEMENT BACKTEST")
    print(f"{'='*60}")

    # Simulate 100 betting opportunities
    initial_bankroll = 10000.0
    results = {
        'with_risk': {'bankroll': initial_bankroll, 'wins': 0, 'losses': 0},
        'without_risk': {'bankroll': initial_bankroll, 'wins': 0, 'losses': 0}
    }

    # Generate sample signals
    import random
    random.seed(42)

    for _ in range(100):
        true_prob = random.uniform(0.45, 0.65)
        offered_odds = 1 / (true_prob - random.uniform(0, 0.1))

        # With risk management (Kelly, capped at 5%)
        kelly_numerator = (true_prob * offered_odds - 1)
        denom = max(offered_odds - 1, 1e-9)
        kelly_fraction = max(0.0, 0.25 * (kelly_numerator / denom))
        kelly_stake = min(initial_bankroll * 0.05, results['with_risk']['bankroll'] * kelly_fraction)

        # Without risk management (flat stake)
        flat_stake = initial_bankroll * 0.02

        won = random.random() < true_prob

        if kelly_stake > 0:
            if won:
                results['with_risk']['bankroll'] += kelly_stake * (offered_odds - 1)
                results['with_risk']['wins'] += 1
            else:
                results['with_risk']['bankroll'] -= kelly_stake
                results['with_risk']['losses'] += 1

        if results['without_risk']['bankroll'] > flat_stake:
            if won:
                results['without_risk']['bankroll'] += flat_stake * (offered_odds - 1)
                results['without_risk']['wins'] += 1
            else:
                results['without_risk']['bankroll'] -= flat_stake
                results['without_risk']['losses'] += 1

    print("\n\ud83d\udcc8 Results After 100 Bets:")
    print(f"\nWith Risk Management (Kelly):")
    print(f"  Final Bankroll: ${results['with_risk']['bankroll']:.2f}")
    print(f"  ROI: {((results['with_risk']['bankroll']/initial_bankroll - 1) * 100):.1f}%")
    total_with = results['with_risk']['wins'] + results['with_risk']['losses']
    print(f"  Win Rate: {results['with_risk']['wins']}/{total_with}")

    print(f"\nWithout Risk Management (Flat):")
    print(f"  Final Bankroll: ${results['without_risk']['bankroll']:.2f}")
    print(f"  ROI: {((results['without_risk']['bankroll']/initial_bankroll - 1) * 100):.1f}%")
    total_wo = results['without_risk']['wins'] + results['without_risk']['losses']
    print(f"  Win Rate: {results['without_risk']['wins']}/{total_wo}")


if __name__ == "__main__":
    print("Starting Risk Management System demo...")
    # Run main example
    asyncio.run(main())
    # Run advanced example
    # asyncio.run(advanced_example())
    # Run backtest
    # asyncio.run(backtest_risk_management())
 