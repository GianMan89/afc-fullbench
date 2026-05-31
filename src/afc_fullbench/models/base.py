"""Base estimator interface for full-episode AFC classifiers.

The classes in :mod:`afc_fullbench.models` intentionally implement a small
scikit-learn-like API.  Each model is fitted on complete alarm-flood episodes
and returns predictions for complete held-out episodes.  No online prefixes,
perturbations, or robustness-specific state are used in this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AFCClassifier(ABC):
    """Minimal estimator API used by the full-episode benchmark runner.

    Concrete classifiers must implement :meth:`fit` and either
    :meth:`predict_proba` or :meth:`predict`.  The default :meth:`predict`
    method converts class probabilities to class labels using ``self.classes_``.
    """

    name: str = "AFCClassifier"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AFCClassifier":
        """Fit the classifier on complete alarm-flood episodes.

        Parameters
        ----------
        X:
            Binary alarm-state tensor with shape
            ``(n_episodes, n_alarm_tags, n_time_steps)``.
        y:
            Integer class labels with shape ``(n_episodes,)``.
        """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities with columns aligned to ``self.classes_``.

        Subclasses without native probabilities may override :meth:`predict`
        directly instead.  Distance-based classifiers use a stable softmax over
        negative distances to obtain probability-like scores.
        """

        raise NotImplementedError(f"{type(self).__name__} does not implement predict_proba().")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for complete alarm-flood episodes."""

        if not hasattr(self, "classes_"):
            raise AttributeError("model is not fitted; missing classes_ attribute")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(proba, axis=1)]

    def get_params(self) -> dict[str, Any]:
        """Return serializable model parameters for result metadata."""

        return {}


def ensure_3d(X: np.ndarray) -> np.ndarray:
    """Validate and return a 3-D alarm-series tensor.

    A single episode with shape ``(n_tags, n_time_steps)`` is promoted to one
    sample.  The function does not force a dtype so that downstream code can
    choose either ``float32`` or ``float64`` depending on backend requirements.
    """

    arr = np.asarray(X)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_tags, n_time_steps)")
    return arr


def as_binary_3d(X: np.ndarray, *, dtype: type = np.float32) -> np.ndarray:
    """Return ``X`` as a binary tensor with shape ``(n, tags, time)``."""

    arr = ensure_3d(X)
    return (arr > 0).astype(dtype, copy=False)


def stable_softmax_from_scores(scores: np.ndarray, *, higher_is_better: bool = True) -> np.ndarray:
    """Convert arbitrary scores or distances to stable probability-like values.

    Parameters
    ----------
    scores:
        Two-dimensional score matrix with rows as samples and columns as classes.
    higher_is_better:
        If ``False``, scores are interpreted as distances and negated before the
        softmax transformation.
    """

    logits = np.asarray(scores, dtype=float)
    if logits.ndim == 1:
        logits = logits[:, None]
    if not higher_is_better:
        logits = -logits
    logits = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    denom = exp_logits.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp_logits / denom
