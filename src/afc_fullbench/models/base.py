"""Base classes for AFC classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AFCClassifier(ABC):
    """Minimal estimator API used by the benchmark runner."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AFCClassifier":
        """Fit the classifier on full alarm-flood episodes."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for full alarm-flood episodes."""

    def get_params(self) -> dict:
        """Return serializable model parameters."""
        return {}
