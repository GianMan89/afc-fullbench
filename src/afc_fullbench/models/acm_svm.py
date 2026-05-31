"""Alarm coactivation matrix with support vector machine classifier."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from afc_fullbench.models.base import AFCClassifier
from afc_fullbench.representations import coactivation_matrix_features


class ACMSVM(AFCClassifier):
    """Series-based alarm coactivation matrix with SVM classifier."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        class_weight: str | dict | None = None,
        random_state: int = 0,
    ):
        self.C = float(C)
        self.kernel = str(kernel)
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ACMSVM":
        Z = coactivation_matrix_features(X, include_diagonal=True, normalize=True)
        clf = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model_ = make_pipeline(StandardScaler(), clf)
        self.model_.fit(Z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = coactivation_matrix_features(X, include_diagonal=True, normalize=True)
        return self.model_.predict(Z)

    def get_params(self) -> dict:
        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }
