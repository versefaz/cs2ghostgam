import numpy as np
import pandas as pd
from typing import Dict


def simple_backtest(y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.55) -> Dict:
    bet_mask = y_proba > threshold
    if np.sum(bet_mask) == 0:
        return {'roi': 0.0, 'bets': 0}
    returns = np.where(y_true[bet_mask] == 1, 1.0, -1.0)
    roi = float((np.sum(returns) / np.sum(bet_mask)) * 100)
    return {'roi': roi, 'bets': int(np.sum(bet_mask))}
