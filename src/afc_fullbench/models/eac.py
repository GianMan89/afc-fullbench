"""Exponentially attenuated components 1-nearest-neighbor classifier."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from afc_fullbench.models.base import AFCClassifier
from afc_fullbench.representations import eac_features


class EAC1NN(AFCClassifier):
    """Sequence-based exponentially attenuated component 1-NN classifier."""

    def __init__(self, *, attenuation: float = 0.01, metric: str = "euclidean"):
        self.attenuation = float(attenuation)
        self.metric = str(metric)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EAC1NN":
        Z = eac_features(X, attenuation=self.attenuation)
        self.model_ = KNeighborsClassifier(n_neighbors=1, metric=self.metric)
        self.model_.fit(Z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = eac_features(X, attenuation=self.attenuation)
        return self.model_.predict(Z)

    def get_params(self) -> dict:
        return {"attenuation": self.attenuation, "metric": self.metric}
