"""Weighted dissimilarity template classifier for complete episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afc_fullbench.models.base import AFCClassifier, ensure_3d, stable_softmax_from_scores
from afc_fullbench.representations import active_set_vector


@dataclass
class WDI1NN(AFCClassifier):
    """Set-based weighted dissimilarity classifier using class templates.

    The implementation follows the WDI-style template construction used in the
    AFC-RobustBench baseline: each class is represented by a binary alarm-set
    template obtained from the training episodes, and class-specific weights are
    derived from within-class and between-class alarm frequencies.
    """

    template_threshold: float = 0.5
    name: str = "WDI-1NN"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WDI1NN":
        """Fit class templates and class-specific alarm weights."""

        X = ensure_3d(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        active = np.stack([active_set_vector(sample) for sample in X])
        self.n_tags_ = active.shape[1]
        self.templates_: dict[int, np.ndarray] = {}
        self.weights_: dict[int, np.ndarray] = {}

        class_freq: dict[int, np.ndarray] = {}
        for cls in self.classes_:
            freq = active[y == cls].mean(axis=0)
            template = (freq > float(self.template_threshold)).astype(float)
            class_freq[int(cls)] = freq
            self.templates_[int(cls)] = template

        alpha: dict[int, np.ndarray] = {}
        for cls in self.classes_:
            template = self.templates_[int(cls)]
            freq = class_freq[int(cls)]
            alpha[int(cls)] = (freq * template) + ((1.0 - freq) * (1.0 - template))

        for cls in self.classes_:
            if len(self.classes_) == 1:
                beta = np.zeros(self.n_tags_)
            else:
                beta = (sum(alpha.values()) - alpha[int(cls)]) / (len(self.classes_) - 1)
            self.weights_[int(cls)] = (2.0 * alpha[int(cls)] - 1.0) * (1.0 - beta)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax-normalized negative weighted dissimilarities."""

        X = ensure_3d(X)
        active = np.stack([active_set_vector(sample) for sample in X])
        distances = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for cidx, cls in enumerate(self.classes_):
            weights = self.weights_[int(cls)]
            template = self.templates_[int(cls)]
            denom = float(np.sum(np.abs(weights)))
            if denom <= 1e-12:
                distances[:, cidx] = np.mean(np.abs(active - template), axis=1)
            else:
                distances[:, cidx] = np.sum(weights * np.abs(active - template), axis=1) / denom
        return stable_softmax_from_scores(distances, higher_is_better=False)

    def get_params(self) -> dict[str, float]:
        """Return serializable hyperparameters."""

        return {"template_threshold": float(self.template_threshold)}
