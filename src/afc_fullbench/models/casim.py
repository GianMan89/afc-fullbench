"""CASIM-style convolutional-kernel classifier for complete episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import warnings
import inspect

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from afc_fullbench.models.base import AFCClassifier, ensure_3d, stable_softmax_from_scores

Backend = Literal["auto", "sktime", "lite"]


class _RandomConvolutionFeatures:
    """Deterministic random-convolution feature extractor used as fallback.

    The fallback is intentionally lightweight and dependency-free.  It is useful
    for development, tests, and machines without ``sktime``/``numba``.  For
    publication-grade CASIM-style experiments, install the optional ``casim``
    dependencies and use ``backend='auto'`` or ``backend='sktime'``.
    """

    def __init__(self, n_kernels: int = 128, random_state: int = 42) -> None:
        self.n_kernels = int(n_kernels)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray) -> "_RandomConvolutionFeatures":
        """Sample deterministic kernels from the training data shape."""

        X = ensure_3d(X)
        rng = np.random.default_rng(self.random_state)
        _, n_tags, n_time = X.shape
        lengths = np.array([length for length in (3, 5, 7, 9) if length <= max(n_time, 3)])
        if len(lengths) == 0:
            lengths = np.array([min(3, n_time)])
        self.kernels_: list[dict[str, Any]] = []
        for _ in range(self.n_kernels):
            length = int(rng.choice(lengths))
            n_channels = int(rng.integers(1, min(n_tags, 3) + 1))
            channels = rng.choice(n_tags, size=n_channels, replace=False)
            weights = rng.normal(size=length)
            weights = weights - weights.mean()
            norm = np.linalg.norm(weights)
            if norm > 0:
                weights = weights / norm
            dilation = int(rng.integers(1, max(2, n_time // max(length, 1) + 1)))
            self.kernels_.append(
                {"channels": channels, "weights": weights, "dilation": dilation, "length": length}
            )
        return self

    @staticmethod
    def _apply_kernel(sample: np.ndarray, kernel: dict[str, Any]) -> np.ndarray:
        """Apply one dilated random kernel to one multivariate episode."""

        channels = kernel["channels"]
        weights = kernel["weights"]
        dilation = int(kernel["dilation"])
        length = int(kernel["length"])
        idx = np.arange(length) * dilation
        max_start = sample.shape[1] - int(idx[-1])
        signal = sample[channels].mean(axis=0)
        if max_start <= 0:
            usable = min(len(signal), length)
            return np.array([float(np.dot(signal[:usable], weights[:usable]))], dtype=float)
        vals = np.empty(max_start, dtype=float)
        for start in range(max_start):
            vals[start] = float(np.dot(signal[start + idx], weights))
        return vals

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform a batch of episodes to random-convolution features."""

        X = ensure_3d(X).astype(float, copy=False)
        features = np.zeros((X.shape[0], 4 * len(self.kernels_)), dtype=float)
        for i, sample in enumerate(X):
            row = []
            for kernel in self.kernels_:
                vals = self._apply_kernel(sample, kernel)
                row.extend([vals.max(), vals.mean(), vals.min(), np.mean(vals > 0.0)])
            features[i] = row
        return np.nan_to_num(features)


@dataclass(init=False)
class CASIM(AFCClassifier):
    """Series-based convolutional-kernel classifier for complete episodes.

    Parameters mirror the CASIM configuration used in AFC-RobustBench where
    possible.  ``backend='auto'`` first tries the optional ``sktime`` MultiRocket
    transformer and falls back to a deterministic random-convolution feature
    extractor if the optional backend is unavailable or incompatible.

    The sktime backend uses ``float64`` input to avoid numba signature errors
    observed with some ``sktime``/``numba`` combinations when using ``float32``
    arrays.
    """

    num_features: int
    n_estimators: int
    n_jobs_multirocket: int
    random_state: int
    alphas: Any
    backend: Backend
    name: str

    def __init__(
        self,
        *,
        num_features: int = 672,
        n_estimators: int = 1,
        n_jobs_multirocket: int = 1,
        random_state: int = 42,
        alphas: Any = None,
        backend: Backend = "auto",
        # Backward-compatible aliases from earlier AFC-FullBench drafts.
        n_kernels: int | None = None,
        use_sktime_if_available: bool | None = None,
    ) -> None:
        if n_kernels is not None:
            num_features = int(n_kernels)
        if use_sktime_if_available is not None:
            backend = "auto" if use_sktime_if_available else "lite"
        self.num_features = int(num_features)
        self.n_estimators = int(n_estimators)
        self.n_jobs_multirocket = int(n_jobs_multirocket)
        self.random_state = int(random_state)
        self.alphas = alphas
        self.backend = backend
        self.name = "CASIM"

    @staticmethod
    def _prepare_series(X: np.ndarray, *, dtype: type = np.float64) -> np.ndarray:
        """Validate, binarize, and cast episodes for convolutional backends."""

        X = ensure_3d(X)
        if X.shape[2] < 9:
            pad = 9 - X.shape[2]
            X = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")
        return (X > 0).astype(dtype, copy=False)

    def _fit_sktime_backend(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Try fitting the optional sktime MultiRocket backend.

        Returns ``True`` if the backend is successfully fitted.  If the backend
        is unavailable or raises an exception and ``backend='auto'``, ``False`` is
        returned so that the caller can use the lite fallback.
        """

        try:
            from sktime.transformations.panel.rocket import MultiRocketMultivariate
        except Exception as exc:
            if self.backend == "sktime":
                raise ImportError(
                    "CASIM backend='sktime' requested, but sktime could not be imported. "
                    "Install with `pip install -e .[casim]`."
                ) from exc
            return False

        X_backend = self._prepare_series(X, dtype=np.float64)
        alphas = np.logspace(-3, 3, 10) if self.alphas is None else np.asarray(self.alphas, dtype=float)

        try:
            signature = inspect.signature(MultiRocketMultivariate)
            kwargs: dict[str, Any] = {}
            if "num_kernels" in signature.parameters:
                kwargs["num_kernels"] = max(84, int(self.num_features))
            elif "num_features" in signature.parameters:
                kwargs["num_features"] = int(self.num_features)
            if "random_state" in signature.parameters:
                kwargs["random_state"] = int(self.random_state)
            if "n_jobs" in signature.parameters:
                kwargs["n_jobs"] = int(self.n_jobs_multirocket)
            transformer = MultiRocketMultivariate(**kwargs)
            transformer.fit(X_backend)
            features = transformer.transform(X_backend)
            self.transformer_ = transformer
            self.classifier_ = make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifierCV(alphas=alphas),
            )
            self.classifier_.fit(features, y)
            self.classes_ = self.classifier_[-1].classes_
            self.backend_ = "sktime"
            return True
        except Exception as exc:
            if self.backend == "sktime":
                raise RuntimeError(
                    "CASIM sktime backend failed. This is often caused by an "
                    "incompatible sktime/numba combination. Try backend='lite' "
                    "or update sktime and numba."
                ) from exc
            warnings.warn(
                "CASIM sktime backend failed; falling back to deterministic CASIM-lite features. "
                f"Original error: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def _fit_lite_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the dependency-free random-convolution fallback."""

        X_lite = self._prepare_series(X, dtype=float)
        n_kernels = max(8, int(self.num_features) // 8)
        self.extractor_ = _RandomConvolutionFeatures(
            n_kernels=n_kernels,
            random_state=int(self.random_state),
        ).fit(X_lite)
        features = self.extractor_.transform(X_lite)
        alphas = np.logspace(-3, 3, 10) if self.alphas is None else np.asarray(self.alphas, dtype=float)
        base = make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(alphas=alphas),
        )
        # Calibrated probabilities are convenient for a common API; keep the
        # calibration CV small so the fallback remains usable on small datasets.
        y_int = np.asarray(y)
        _, counts = np.unique(y_int, return_counts=True)
        min_class_count = int(counts.min()) if counts.size else 0
        if min_class_count >= 2:
            cv = min(3, min_class_count)
            self.classifier_ = CalibratedClassifierCV(base, cv=cv)
            self.classifier_.fit(features, y_int)
            self.classes_ = self.classifier_.classes_
        else:
            # Very small development datasets may not support internal
            # calibration.  In that case use the ridge classifier directly.
            self.classifier_ = base
            self.classifier_.fit(features, y_int)
            self.classes_ = self.classifier_[-1].classes_
        self.backend_ = "lite"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CASIM":
        """Fit CASIM-style convolutional features and a ridge classifier."""

        X = ensure_3d(X)
        self.train_length_ = max(9, X.shape[2])
        y = np.asarray(y)
        fitted = False
        if self.backend in {"auto", "sktime"}:
            fitted = self._fit_sktime_backend(X, y)
        if not fitted:
            self._fit_lite_backend(X, y)
        return self

    def _pad_to_train_length(self, X: np.ndarray) -> np.ndarray:
        """Pad or truncate episodes to the training horizon used by the backend."""

        X = ensure_3d(X)
        target = getattr(self, "train_length_", X.shape[2])
        if X.shape[2] == target:
            return X
        if X.shape[2] > target:
            return X[:, :, :target]
        pad = target - X.shape[2]
        return np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")

    def _features(self, X: np.ndarray):
        """Return backend-specific convolutional features for prediction."""

        X = self._pad_to_train_length(X)
        if getattr(self, "backend_", None) == "sktime":
            X_backend = self._prepare_series(X, dtype=np.float64)
            return self.transformer_.transform(X_backend)
        X_lite = self._prepare_series(X, dtype=float)
        return self.extractor_.transform(X_lite)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using the fitted ridge or calibrated classifier."""

        return self.classifier_.predict(self._features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability-like class scores.

        The lite backend exposes calibrated probabilities.  The sktime/ridge
        backend is converted through a stable softmax over decision scores.
        """

        features = self._features(X)
        if hasattr(self.classifier_, "predict_proba"):
            try:
                return self.classifier_.predict_proba(features)
            except Exception:
                pass
        scores = self.classifier_.decision_function(features)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return stable_softmax_from_scores(scores, higher_is_better=True)

    def get_params(self) -> dict[str, Any]:
        """Return serializable hyperparameters."""

        return {
            "num_features": int(self.num_features),
            "n_estimators": int(self.n_estimators),
            "n_jobs_multirocket": int(self.n_jobs_multirocket),
            "random_state": int(self.random_state),
            "backend": self.backend,
        }
