"""Representation utilities for full alarm-flood episodes."""

from __future__ import annotations

import numpy as np


def validate_series_tensor(X: np.ndarray) -> np.ndarray:
    """Return ``X`` as a binary float tensor with shape ``(n, tags, time)``."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_episodes, n_tags, n_time_steps), got {X.shape}.")
    return (X > 0).astype(np.float32, copy=False)


def activation_events(series: np.ndarray) -> list[tuple[int, int]]:
    """Extract activation events from one binary alarm-series episode.

    Returns a list of ``(tag_index, time_index)`` tuples sorted by time and then tag.
    A tag active at the first time step is treated as an activation at time zero.
    """
    S = (np.asarray(series) > 0).astype(np.int8)
    if S.ndim != 2:
        raise ValueError("Expected one episode with shape (n_tags, n_time_steps).")
    padded = np.pad(S, ((0, 0), (1, 0)), mode="constant")
    rising = np.diff(padded, axis=1) > 0
    tags, times = np.where(rising)
    order = np.lexsort((tags, times))
    return [(int(tags[i]), int(times[i])) for i in order]


def alarm_set_features(X: np.ndarray) -> np.ndarray:
    """Binary alarm-set features indicating whether each tag activated during the episode."""
    X = validate_series_tensor(X)
    features = []
    for episode in X:
        events = activation_events(episode)
        vec = np.zeros(episode.shape[0], dtype=np.float32)
        for tag, _ in events:
            vec[tag] = 1.0
        features.append(vec)
    return np.vstack(features)


def activation_count_features(X: np.ndarray, *, log_scale: bool = True) -> np.ndarray:
    """Bag-of-activation-count features for each alarm tag."""
    X = validate_series_tensor(X)
    features = []
    for episode in X:
        vec = np.zeros(episode.shape[0], dtype=np.float32)
        for tag, _ in activation_events(episode):
            vec[tag] += 1.0
        if log_scale:
            vec = np.log1p(vec)
        features.append(vec)
    return np.vstack(features)


def eac_features(X: np.ndarray, *, attenuation: float = 0.01) -> np.ndarray:
    """Exponentially attenuated activation-component features.

    Each activation contributes ``exp(-attenuation * rank)`` to its alarm tag,
    where ``rank`` is the position of the activation in the full episode sequence.
    """
    X = validate_series_tensor(X)
    features = []
    for episode in X:
        vec = np.zeros(episode.shape[0], dtype=np.float32)
        for rank, (tag, _) in enumerate(activation_events(episode)):
            vec[tag] += np.exp(-float(attenuation) * rank)
        features.append(vec)
    return np.vstack(features)


def coactivation_matrix_features(
    X: np.ndarray,
    *,
    include_diagonal: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Flattened alarm coactivation matrix features.

    For each episode, the coactivation matrix is ``S @ S.T`` where ``S`` is the
    binary alarm-state matrix with shape ``(n_tags, n_time_steps)``.
    """
    X = validate_series_tensor(X)
    features = []
    for episode in X:
        T = max(1, episode.shape[1])
        mat = episode @ episode.T
        if normalize:
            mat = mat / float(T)
        k = 0 if include_diagonal else 1
        tri = np.triu_indices(mat.shape[0], k=k)
        features.append(mat[tri].astype(np.float32, copy=False))
    return np.vstack(features)


def flatten_series_features(X: np.ndarray) -> np.ndarray:
    """Flatten full binary alarm series for generic time-series classifiers."""
    X = validate_series_tensor(X)
    return X.reshape(X.shape[0], -1)
