from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def time_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, val_size: float = 0.2,
               time_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Time-aware split: train/val/test by chronological order"""
    df_sorted = df.sort_values(time_col)
    y = df_sorted[target_col]
    X = df_sorted.drop(columns=[target_col])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, shuffle=False)
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - relative_val_size, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test
