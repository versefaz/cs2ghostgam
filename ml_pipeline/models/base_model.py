from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Any:
        ...
