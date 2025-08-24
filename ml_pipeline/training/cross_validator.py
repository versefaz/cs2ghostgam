from sklearn.model_selection import TimeSeriesSplit
from typing import Dict
import numpy as np
import pandas as pd


def time_series_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
    return {'scores': scores, 'mean_score': float(np.mean(scores))}
