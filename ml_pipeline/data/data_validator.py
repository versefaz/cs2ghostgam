from typing import Dict
import pandas as pd


def basic_quality_checks(df: pd.DataFrame, required_cols: Dict[str, str]) -> Dict[str, int]:
    """Check missing columns and NaNs"""
    issues = {'missing_columns': 0, 'nans': 0}
    for col, dtype in required_cols.items():
        if col not in df.columns:
            issues['missing_columns'] += 1
        else:
            issues['nans'] += int(df[col].isna().sum())
    return issues
