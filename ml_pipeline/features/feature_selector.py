from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from typing import Tuple


def select_top_k(X: pd.DataFrame, y: pd.Series, k: int = 50) -> Tuple[pd.DataFrame, list]:
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    cols = X.columns[selector.get_support(indices=True)]
    return pd.DataFrame(X_new, columns=cols), list(cols)
