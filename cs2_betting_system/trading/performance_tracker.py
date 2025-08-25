import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datetime import datetime


class PerformanceTracker:
    def __init__(self, reports_dir: str = 'reports'):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def summarize(self, positions: List[Dict], initial_balance: float, balance: float) -> Dict:
        completed = [b for b in positions if b.get('status') in ['won', 'lost']]
        won = [b for b in completed if b.get('status') == 'won']
        total_staked = sum(b.get('amount', 0) for b in completed)
        total_returns = sum(b.get('returns', 0) for b in completed)
        win_rate = (len(won) / len(completed)) if completed else 0
        roi = ((total_returns - total_staked) / total_staked * 100) if total_staked > 0 else 0
        profit = balance - initial_balance
        return {
            'total_bets': len(positions),
            'completed': len(completed),
            'won': len(won),
            'win_rate': win_rate,
            'roi': roi,
            'profit': profit,
            'final_balance': balance,
        }

    def save_report(self, positions: List[Dict], summary: Dict):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(positions)
        df.to_csv(self.reports_dir / f'paper_trading_{ts}.csv', index=False)
        with open(self.reports_dir / f'summary_{ts}.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def generate_report(self, positions: List[Dict] = None, initial_balance: float = 0, balance: float = 0):
        if positions is None:
            return
        summary = self.summarize(positions, initial_balance, balance)
        self.save_report(positions, summary)
