"""Model factory for configuration-driven full-episode AFC experiments."""

from __future__ import annotations

from typing import Any

from afc_fullbench.models.acm_svm import ACMSVM
from afc_fullbench.models.casim import CASIM
from afc_fullbench.models.eac import EAC1NN
from afc_fullbench.models.jaccard import Jaccard1NN
from afc_fullbench.models.mbw_lr import MBWLogisticRegression
from afc_fullbench.models.wdi import WDI1NN

MODEL_REGISTRY = {
    "wdi_1nn": WDI1NN,
    "WDI-1NN": WDI1NN,
    "jac_1nn": Jaccard1NN,
    "JAC-1NN": Jaccard1NN,
    "eac_1nn": EAC1NN,
    "EAC-1NN": EAC1NN,
    "mbw_lr": MBWLogisticRegression,
    "MBW-LR": MBWLogisticRegression,
    "acm_svm": ACMSVM,
    "ACM-SVM": ACMSVM,
    "casim": CASIM,
    "CASIM": CASIM,
}

DISPLAY_NAMES = {
    "wdi_1nn": "WDI-1NN",
    "WDI-1NN": "WDI-1NN",
    "jac_1nn": "JAC-1NN",
    "JAC-1NN": "JAC-1NN",
    "eac_1nn": "EAC-1NN",
    "EAC-1NN": "EAC-1NN",
    "mbw_lr": "MBW-LR",
    "MBW-LR": "MBW-LR",
    "acm_svm": "ACM-SVM",
    "ACM-SVM": "ACM-SVM",
    "casim": "CASIM",
    "CASIM": "CASIM",
}

# Parameter names used only by AFC-RobustBench online prefix wrappers.  They are
# intentionally ignored here because AFC-FullBench trains one model per complete
# episode, not prefix-specific classifiers.
_ONLINE_ONLY_KEYS = {
    "training_mode",
    "training_strategy",
    "prefix_grid",
    "prefix_reference",
    "prefix_train_reference",
    "prefix_selection",
    "prefix_train_horizon",
    "prefix_min_time_steps",
    "include_full_prefix",
}


def _clean_full_episode_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove online-only parameters from AFC-RobustBench configurations."""

    return {key: value for key, value in params.items() if key not in _ONLINE_ONLY_KEYS}


def make_model(name: str, params: dict[str, Any] | None = None):
    """Instantiate a full-episode AFC model by registry name.

    The factory accepts the same baseline method names used in AFC-RobustBench.
    If a copied configuration contains online-prefix wrapper keys, those keys are
    removed because they have no meaning for full-episode classification.
    """

    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model name: {name}; available: {available_models()}")
    clean_params = _clean_full_episode_params({} if params is None else dict(params))
    return MODEL_REGISTRY[name](**clean_params)


def display_name(name: str) -> str:
    """Return canonical display name for a registry key."""

    return DISPLAY_NAMES.get(name, str(name))


def available_models() -> list[str]:
    """Return supported registry keys."""

    return sorted(MODEL_REGISTRY)
