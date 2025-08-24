from typing import Any


class ModelServer:
    def __init__(self, model: Any):
        self.model = model

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
