class Backtester:
    """ทดสอบย้อนหลัง 1 ปี (skeleton)"""
    def run_backtest(self, start_date: str, end_date: str):
        initial_bankroll = 10000.0
        current_bankroll = initial_bankroll
        bets = []
        # TODO: implement loading historical matches, predictions, and odds
        roi = (current_bankroll - initial_bankroll) / initial_bankroll
        return { 'roi': roi, 'total_bets': len(bets) }
