"""CASIM-style convolutional-kernel classifier for full episodes."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_random_state

from afc_fullbench.models.base import AFCClassifier
from afc_fullbench.representations import validate_series_tensor


class _RandomConvolutionTransformer:
    """Small deterministic random-convolution feature transformer.

    This fallback is not a drop-in replacement for full MultiRocket/CASIM, but it
    preserves the intended convolutional full-series feature family when optional
    time-series dependencies are unavailable.
    """

    def __init__(self, *, n_kernels: int = 512, random_state: int = 0):
        self.n_kernels = int(n_kernels)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray) -> "_RandomConvolutionTransformer":
        X = validate_series_tensor(X)
        rng = check_random_state(self.random_state)
        n_channels = X.shape[1]
        max_length = X.shape[2]
        lengths = np.array([3, 5, 7, 9], dtype=int)
        lengths = lengths[lengths <= max_length]
        if lengths.size == 0:
            lengths = np.array([max_length], dtype=int)

        self.kernels_ = []
        for _ in range(self.n_kernels):
            length = int(rng.choice(lengths))
            channel = int(rng.randint(0, n_channels))
            weights = rng.normal(size=length).astype(np.float32)
            weights -= weights.mean()
            norm = np.linalg.norm(weights)
            if norm > 0:
                weights /= norm
            self.kernels_.append((channel, weights))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = validate_series_tensor(X)
        features = np.zeros((X.shape[0], 4 * len(self.kernels_)), dtype=np.float32)
        for k, (channel, weights) in enumerate(self.kernels_):
            length = len(weights)
            for i, episode in enumerate(X):
                signal = episode[channel]
                if signal.size < length:
                    conv = np.array([float(np.dot(signal, weights[: signal.size]))], dtype=np.float32)
                else:
                    conv = np.convolve(signal, weights[::-1], mode="valid")
                offset = 4 * k
                features[i, offset] = float(conv.max())
                features[i, offset + 1] = float(conv.min())
                features[i, offset + 2] = float(conv.mean())
                features[i, offset + 3] = float((conv > 0).mean())
        return features


class CASIM(AFCClassifier):
    """Series-based convolutional-kernel classifier for complete alarm episodes.

    If ``sktime`` is installed, this class uses ``MultiRocketMultivariate`` as a
    strong convolutional feature extractor. Otherwise it falls back to a compact
    deterministic random-convolution transformer.
    """

    def __init__(
        self,
        *,
        n_kernels: int = 672,
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
        random_state: int = 0,
        use_sktime_if_available: bool = True,
    ):
        self.n_kernels = int(n_kernels)
        self.alphas = tuple(float(a) for a in alphas)
        self.random_state = int(random_state)
        self.use_sktime_if_available = bool(use_sktime_if_available)

    def _make_transformer(self):
        if self.use_sktime_if_available:
            try:
                from sktime.transformations.panel.rocket import MultiRocketMultivariate

                return MultiRocketMultivariate(
                    num_kernels=max(84, self.n_kernels),
                    random_state=self.random_state,
                )
            except Exception:
                pass
        return _RandomConvolutionTransformer(
            n_kernels=self.n_kernels,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CASIM":
        X = validate_series_tensor(X)
        self.transformer_ = self._make_transformer()
        self.transformer_.fit(X)
        Z = self.transformer_.transform(X)
        self.classifier_ = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.asarray(self.alphas, dtype=float)),
        )
        self.classifier_.fit(Z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = validate_series_tensor(X)
        Z = self.transformer_.transform(X)
        return self.classifier_.predict(Z)

    def get_params(self) -> dict:
        return {
            "n_kernels": self.n_kernels,
            "alphas": self.alphas,
            "random_state": self.random_state,
            "use_sktime_if_available": self.use_sktime_if_available,
        }
