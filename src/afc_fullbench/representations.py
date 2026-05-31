"""Representation utilities for complete alarm-flood episodes.

The benchmark loads binary alarm-state matrices with shape
``(n_tags, n_time_steps)``.  The functions in this module derive the three
standard AFC views used by the implemented classifiers:

* alarm sets: which tags activated at least once;
* alarm activation sequences: rising edges ordered by time and tag index;
* alarm series: binary active-state trajectories over time.
"""

from __future__ import annotations

import numpy as np

from afc_fullbench.models.base import as_binary_3d, ensure_3d


def validate_series_tensor(X: np.ndarray, *, dtype: type = np.float32) -> np.ndarray:
    """Return ``X`` as a binary tensor with shape ``(n, tags, time)``."""

    return as_binary_3d(X, dtype=dtype)


def activation_sequence_from_series(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return ordered activation events ``(tag_index, time_index)``.

    A tag active at the first time step is interpreted as an activation at
    ``time_index == 0``.  Subsequent activations are rising edges in the binary
    alarm-state trajectory.  Events are sorted by time and then by tag index.
    """

    x = (np.asarray(matrix) > 0).astype(np.int8, copy=False)
    if x.ndim != 2:
        raise ValueError("matrix must have shape (n_tags, n_time_steps)")

    events: list[tuple[int, int]] = []
    n_tags, _ = x.shape
    for tag in range(n_tags):
        if x[tag, 0] == 1:
            events.append((tag, 0))
        rising = np.where(np.diff(x[tag].astype(np.int8)) == 1)[0] + 1
        for time_idx in rising:
            events.append((tag, int(time_idx)))
    events.sort(key=lambda item: (item[1], item[0]))
    return events


def active_set_vector(matrix: np.ndarray) -> np.ndarray:
    """Return a binary vector indicating which tags are active at least once."""

    x = (np.asarray(matrix) > 0).astype(np.int8, copy=False)
    if x.ndim != 2:
        raise ValueError("matrix must have shape (n_tags, n_time_steps)")
    return (x.max(axis=1) > 0).astype(np.int8)


def alarm_set_features(X: np.ndarray) -> np.ndarray:
    """Return alarm-set features for a batch of complete episodes."""

    X = ensure_3d(X)
    return np.stack([active_set_vector(sample) for sample in X]).astype(np.float32)


def activation_count_features(X: np.ndarray, *, log_scale: bool = False) -> np.ndarray:
    """Return bag-of-activation-count features by alarm tag.

    This helper is used for diagnostics and simple baselines.  The MBW-LR
    implementation uses its own TF--IDF and first-time weighting, following the
    AFC-RobustBench implementation.
    """

    X = ensure_3d(X)
    features = []
    for sample in X:
        row = np.zeros(sample.shape[0], dtype=float)
        for tag, _ in activation_sequence_from_series(sample):
            row[tag] += 1.0
        if log_scale:
            row = np.log1p(row)
        features.append(row)
    return np.asarray(features, dtype=np.float32)


def eac_features(
    X: np.ndarray,
    *,
    attenuation: float = 0.01,
    time_scale: str = "event_index",
    normalize: bool = True,
) -> np.ndarray:
    """Return exponentially attenuated component features.

    Each activation contributes ``exp(-attenuation * t)`` to its tag component,
    where ``t`` is either the activation-event index or the discrete time index.
    """

    X = ensure_3d(X)
    features = []
    for sample in X:
        row = np.zeros(sample.shape[0], dtype=float)
        for event_index, (tag, time_idx) in enumerate(activation_sequence_from_series(sample)):
            t = event_index if time_scale == "event_index" else time_idx
            row[tag] += float(np.exp(-float(attenuation) * t))
        if normalize:
            norm = np.linalg.norm(row)
            if norm > 0:
                row = row / norm
        features.append(row)
    return np.asarray(features, dtype=np.float32)


def flatten_series_features(X: np.ndarray) -> np.ndarray:
    """Flatten full binary alarm series into one vector per episode."""

    X = validate_series_tensor(X)
    return X.reshape(X.shape[0], -1)
