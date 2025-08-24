from typing import Dict, Any


class InMemoryModelRegistry:
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def register(self, name: str, model: Any):
        self._store[name] = model

    def get(self, name: str) -> Any:
        return self._store.get(name)
