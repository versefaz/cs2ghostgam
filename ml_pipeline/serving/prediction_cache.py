from typing import Dict, Any


class PredictionCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self._cache[key] = value

    def get(self, key: str) -> Any:
        return self._cache.get(key)
