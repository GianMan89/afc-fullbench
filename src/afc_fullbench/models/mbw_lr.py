"""Modified bag-of-words logistic-regression classifier."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from afc_fullbench.models.base import AFCClassifier
from afc_fullbench.representations import activation_count_features


class MBWLogisticRegression(AFCClassifier):
    """Sequence-based bag-of-activation-count logistic regression classifier."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 2000,
        class_weight: str | dict | None = None,
        random_state: int = 0,
    ):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MBWLogisticRegression":
        Z = activation_count_features(X, log_scale=True)
        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model_ = make_pipeline(StandardScaler(), clf)
        self.model_.fit(Z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = activation_count_features(X, log_scale=True)
        return self.model_.predict(Z)

    def get_params(self) -> dict:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }
