#!/usr/bin/env python3
import asyncio
import sys
import warnings

from trading.paper_trading_bot import PaperTradingBot
from trading.signal_generator import SignalGenerator
from trading.performance_tracker import PerformanceTracker

warnings.filterwarnings('ignore')


def print_banner():
    banner = (
        "\n"
        "╔═══════════════════════════════════════════════════════╗\n"
        "║     CS2 BETTING PAPER TRADING SYSTEM V1.0            ║\n"
        "║     Win Rate Target: 80%+ | ROI Target: 15%+         ║\n"
        "╚═══════════════════════════════════════════════════════╝\n"
    )
    print(banner)


async def main():
    print_banner()
    print("🔍 Checking system requirements...")

    print("🚀 Initializing components...")
    bot = PaperTradingBot(initial_balance=10000)
    signal_gen = SignalGenerator()
    tracker = PerformanceTracker()
    print("✅ All components initialized successfully!")
    print("-" * 60)

    try:
        await asyncio.get_event_loop().run_in_executor(None, bot.run_simulation)
    except KeyboardInterrupt:
        print("\n🛑 Gracefully shutting down...")
        bot.final_report()
        tracker.generate_report(bot.positions, bot.initial_balance, bot.balance)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        try:
            bot.scraper.cleanup()
        except Exception:
            pass
        print("👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
