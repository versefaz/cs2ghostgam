import pickle
from pathlib import Path
from typing import Any, Dict


class FileFeatureStore:
    def __init__(self, base_dir: str = ".features_cache"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, obj: Any):
        with open(self.base / f"{key}.pkl", "wb") as f:
            pickle.dump(obj, f)

    def load(self, key: str) -> Any:
        with open(self.base / f"{key}.pkl", "rb") as f:
            return pickle.load(f)
