"""Jaccard 1-nearest-neighbor classifier."""

from __future__ import annotations

import numpy as np
from afc_fullbench.models.base import AFCClassifier
from afc_fullbench.representations import alarm_set_features


class Jaccard1NN(AFCClassifier):
    """Set-based Jaccard-distance 1-nearest-neighbor classifier."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Jaccard1NN":
        self.X_train_ = alarm_set_features(X).astype(bool, copy=False)
        self.y_train_ = np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "X_train_"):
            raise RuntimeError("Jaccard1NN has not been fitted yet.")
        Z = alarm_set_features(X).astype(bool, copy=False)
        preds = []
        for row in Z:
            inter = np.logical_and(self.X_train_, row).sum(axis=1)
            union = np.logical_or(self.X_train_, row).sum(axis=1)
            sim = np.divide(inter, union, out=np.ones_like(inter, dtype=float), where=union > 0)
            preds.append(self.y_train_[int(np.argmax(sim))])
        return np.asarray(preds)
