"""Weighted dissimilarity 1-nearest-neighbor classifier."""

from __future__ import annotations

import numpy as np
from afc_fullbench.models.base import AFCClassifier
from afc_fullbench.representations import alarm_set_features


class WDI1NN(AFCClassifier):
    """Set-based weighted dissimilarity 1-nearest-neighbor classifier.

    The implementation uses inverse document frequency style tag weights fitted
    on the training episodes and a weighted symmetric-difference distance.
    """

    def __init__(self, *, epsilon: float = 1e-9):
        self.epsilon = float(epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WDI1NN":
        Z = alarm_set_features(X)
        n = Z.shape[0]
        df = Z.sum(axis=0)
        self.weights_ = np.log((n + 1.0) / (df + 1.0)) + 1.0
        self.X_train_ = Z.astype(np.float32, copy=False)
        self.y_train_ = np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "X_train_"):
            raise RuntimeError("WDI1NN has not been fitted yet.")
        Z = alarm_set_features(X)
        preds = []
        for row in Z:
            diff = np.abs(self.X_train_ - row) * self.weights_
            denom = ((self.X_train_ + row) > 0).astype(np.float32) * self.weights_
            dist = diff.sum(axis=1) / np.maximum(denom.sum(axis=1), self.epsilon)
            preds.append(self.y_train_[int(np.argmin(dist))])
        return np.asarray(preds)

    def get_params(self) -> dict:
        return {"epsilon": self.epsilon}
