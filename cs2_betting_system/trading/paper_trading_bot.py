import json
import time
from datetime import datetime
from typing import List, Dict

import redis

from config import settings
from utils.kelly_criterion import KellyCriterion


class PaperTradingBot:
    def __init__(self, initial_balance: float = settings.INITIAL_BALANCE):
        self.balance = float(initial_balance)
        self.initial_balance = float(initial_balance)
        self.positions: List[Dict] = []
        self.completed_bets: List[Dict] = []
        self.redis = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        self.kelly = KellyCriterion()

        from scrapers.live_match_scraper import LiveMatchScraper
        from models.prediction_model import PredictionModel

        self.scraper = LiveMatchScraper()
        self.model = PredictionModel()

    def calculate_kelly(self, win_prob: float, odds: float) -> float:
        return self.kelly.calculate(win_prob, odds, self.balance)

    def run_simulation(self):
        print("🚀 Starting Paper Trading Bot")
        print(f"💰 Initial Balance: ${self.initial_balance}")
        print("-" * 50)

        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n🔄 Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # 1. Scrape
                print("📊 Scraping matches...")
                live_matches = self.scraper.scrape_all_sources()
                print(f"   Found {len(live_matches)} matches")

                # 2. Predict
                if live_matches:
                    predictions = self.model.batch_predict(live_matches)
                    print(f"   Generated {len(predictions)} predictions")

                    # 3. Evaluate and place bets
                    for pred in predictions:
                        if self.should_bet(pred):
                            self.place_paper_bet(pred)

                # 4. Update results
                self.update_results()

                # 5. Log performance
                self.log_performance()

                # 6. Save state
                self.save_state()

                print(f"💤 Sleeping for {settings.SCRAPE_INTERVAL} seconds...")
                time.sleep(settings.SCRAPE_INTERVAL)

            except KeyboardInterrupt:
                print("\n🛑 Stopping paper trading...")
                self.final_report()
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(60)

    def should_bet(self, prediction: Dict) -> bool:
        if float(prediction.get('confidence', 0)) < settings.MIN_CONFIDENCE:
            return False
        if float(prediction.get('value', 0)) < settings.MIN_VALUE:
            return False

        # Daily limit
        today = datetime.now().date()
        today_bets = [b for b in self.positions if b['timestamp'].date() == today]
        if len(today_bets) >= settings.MAX_BETS_PER_DAY:
            return False

        # Exposure
        current_exposure = sum(b['amount'] for b in self.positions if b['status'] == 'pending')
        if current_exposure > self.balance * settings.MAX_EXPOSURE:
            return False

        # Already bet
        if any(b['match_id'] == prediction.get('match_id') for b in self.positions):
            return False

        return True

    def place_paper_bet(self, prediction: Dict):
        kelly_size = self.calculate_kelly(prediction.get('win_prob', 0.0), prediction.get('odds', 2.0))
        bet_amount = min(self.balance * kelly_size, self.balance * settings.MAX_BET_SIZE)
        if bet_amount < 10:
            return

        bet = {
            'bet_id': f"BET_{datetime.now().timestamp()}",
            'match_id': prediction.get('match_id'),
            'team': prediction.get('team'),
            'amount': round(float(bet_amount), 2),
            'odds': float(prediction.get('odds', 2.0)),
            'confidence': float(prediction.get('confidence', 0.0)),
            'expected_value': float(prediction.get('value', 0.0)),
            'timestamp': datetime.now(),
            'status': 'pending',
        }

        self.balance -= bet['amount']
        self.positions.append(bet)
        self.redis.lpush('paper_bets', json.dumps(bet, default=str))

        print("\n💸 NEW BET PLACED:")
        print(f"   Team: {bet['team']}")
        print(f"   Odds: {bet['odds']:.2f}")
        print(f"   Amount: ${bet['amount']:.2f}")
        print(f"   Confidence: {bet['confidence']:.1%}")
        print(f"   EV: {bet['expected_value']:.2f}")
        print(f"   Balance: ${self.balance:.2f}")

    def update_results(self):
        import random
        for bet in self.positions:
            if bet['status'] == 'pending':
                if random.random() < 0.1:  # match finished
                    if random.random() < bet['confidence']:
                        bet['status'] = 'won'
                        bet['returns'] = bet['amount'] * bet['odds']
                        self.balance += bet['returns']
                        print(f"✅ BET WON: {bet['team']} +${bet['returns']:.2f}")
                    else:
                        bet['status'] = 'lost'
                        bet['returns'] = 0.0
                        print(f"❌ BET LOST: {bet['team']} -${bet['amount']:.2f}")
                    bet['completed_at'] = datetime.now()
                    self.completed_bets.append(bet)

    def log_performance(self):
        total_bets = len(self.positions)
        completed = [b for b in self.positions if b['status'] in ['won', 'lost']]
        won = [b for b in completed if b['status'] == 'won']
        if completed:
            win_rate = len(won) / len(completed)
            total_staked = sum(b['amount'] for b in completed)
            total_returns = sum(b.get('returns', 0) for b in completed)
            roi = ((total_returns - total_staked) / total_staked * 100) if total_staked > 0 else 0
            profit = self.balance - self.initial_balance
            print("\n📊 PERFORMANCE UPDATE:")
            print(f"   Total Bets: {total_bets}")
            print(f"   Completed: {len(completed)}")
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   ROI: {roi:.1f}%")
            print(f"   P&L: ${profit:.2f}")
            print(f"   Current Balance: ${self.balance:.2f}")

    def save_state(self):
        state = {
            'balance': self.balance,
            'positions': self.positions,
            'timestamp': datetime.now().isoformat(),
        }
        self.redis.set('paper_trading_state', json.dumps(state, default=str))

    def final_report(self):
        print("\n" + "=" * 60)
        print("📈 FINAL PAPER TRADING REPORT")
        print("=" * 60)
        completed = [b for b in self.positions if b['status'] in ['won', 'lost']]
        won = [b for b in completed if b['status'] == 'won']
        if completed:
            win_rate = len(won) / len(completed)
            total_staked = sum(b['amount'] for b in completed)
            total_returns = sum(b.get('returns', 0) for b in completed)
            roi = ((total_returns - total_staked) / total_staked * 100) if total_staked > 0 else 0
            profit = self.balance - self.initial_balance
            print(f"Total Bets Placed: {len(self.positions)}")
            print(f"Bets Completed: {len(completed)}")
            print(f"Bets Won: {len(won)}")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Total Staked: ${total_staked:.2f}")
            print(f"Total Returns: ${total_returns:.2f}")
            print(f"ROI: {roi:.1f}%")
            print(f"Net Profit: ${profit:.2f}")
            print(f"Final Balance: ${self.balance:.2f}")
            try:
                import pandas as pd
                from pathlib import Path
                Path('reports').mkdir(exist_ok=True)
                pd.DataFrame(self.positions).to_csv(
                    f'reports/paper_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    index=False
                )
                print("\n💾 Report saved to reports/")
            except Exception:
                pass
