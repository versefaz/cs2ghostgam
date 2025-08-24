from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from typing import Dict


def compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else 0,
    }
