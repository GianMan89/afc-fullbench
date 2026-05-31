"""Alarm coactivation-matrix features with support vector machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from afc_fullbench.models.base import AFCClassifier, ensure_3d


@dataclass
class ACMSVM(AFCClassifier):
    """Series-based alarm coactivation matrix with SVM classifier.

    Features are pairwise Jaccard coactivation values between alarm-state
    trajectories, using the upper triangular part of the coactivation matrix.
    """

    C: float = 1.0
    kernel: str = "rbf"
    gamma: str | float = "scale"
    probability: bool = True
    class_weight: str | dict | None = None
    random_state: int = 42
    name: str = "ACM-SVM"

    @staticmethod
    def coactivation_features(X: np.ndarray) -> np.ndarray:
        """Return flattened pairwise Jaccard coactivation features."""

        X = ensure_3d(X)
        A = (X > 0).astype(np.float64, copy=False)
        n_samples, n_tags, _ = A.shape
        intersections = np.einsum("svt,swt->svw", A, A, optimize=True)
        sums = A.sum(axis=2)
        unions = sums[:, :, None] + sums[:, None, :] - intersections
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(unions > 0, intersections / unions, 0.0)
        iu = np.triu_indices(n_tags, k=1)
        if len(iu[0]) == 0:
            return np.zeros((n_samples, 1), dtype=float)
        return jaccard[:, iu[0], iu[1]]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ACMSVM":
        """Fit an RBF-SVM on coactivation features."""

        features = self.coactivation_features(X)
        self.clf_ = make_pipeline(
            StandardScaler(),
            SVC(
                C=float(self.C),
                kernel=self.kernel,
                gamma=self.gamma,
                probability=bool(self.probability),
                class_weight=self.class_weight,
                random_state=int(self.random_state),
            ),
        )
        self.clf_.fit(features, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return SVM probability estimates."""

        features = self.coactivation_features(X)
        if hasattr(self.clf_[-1], "predict_proba") and self.probability:
            return self.clf_.predict_proba(features)
        scores = self.clf_.decision_function(features)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        from afc_fullbench.models.base import stable_softmax_from_scores

        return stable_softmax_from_scores(scores, higher_is_better=True)

    def get_params(self) -> dict[str, Any]:
        """Return serializable hyperparameters."""

        return {
            "C": float(self.C),
            "kernel": self.kernel,
            "gamma": self.gamma,
            "probability": bool(self.probability),
            "class_weight": self.class_weight,
            "random_state": int(self.random_state),
        }
