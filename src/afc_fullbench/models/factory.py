"""Model factory for configuration-driven experiments."""

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


def make_model(name: str, params: dict[str, Any] | None = None):
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model name: {name}; available: {available_models()}")
    return MODEL_REGISTRY[name](**({} if params is None else dict(params)))


def display_name(name: str) -> str:
    """Return canonical display name for a registry key."""
    return DISPLAY_NAMES.get(name, str(name))


def available_models() -> list[str]:
    """Return supported registry keys."""
    return sorted(MODEL_REGISTRY)
