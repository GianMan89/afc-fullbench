"""Stratified k-fold evaluation for full-episode AFC classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from afc_fullbench.data import AlarmDataset
from afc_fullbench.metrics import classification_metrics
from afc_fullbench.models.factory import display_name, make_model


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one model."""

    name: str
    params: dict[str, Any]
    display_name: str | None = None


def _as_model_configs(configs: list[dict[str, Any]]) -> list[ModelConfig]:
    out: list[ModelConfig] = []
    for item in configs:
        if "name" not in item:
            raise ValueError(f"Model entry is missing `name`: {item}")
        name = str(item["name"])
        params = dict(item.get("params", {}))
        out.append(
            ModelConfig(
                name=name,
                params=params,
                display_name=item.get("display_name", display_name(name)),
            )
        )
    return out


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std metric summary by method."""
    metric_cols = [
        c
        for c in fold_metrics.columns
        if c not in {"fold", "method", "train_size", "test_size", "fit_seconds", "predict_seconds"}
        and pd.api.types.is_numeric_dtype(fold_metrics[c])
    ]
    rows = []
    for method, group in fold_metrics.groupby("method", sort=False):
        row: dict[str, Any] = {"method": method}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_cross_validation(
    dataset: AlarmDataset,
    *,
    model_configs: list[dict[str, Any]] | list[ModelConfig],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run stratified k-fold full-episode classification.

    All models are trained on complete training episodes and tested on complete
    held-out episodes. No online prefixes, perturbations, or robustness scoring
    are applied.
    """

    if not model_configs:
        raise ValueError("At least one model configuration is required.")

    configs = [m if isinstance(m, ModelConfig) else None for m in model_configs]
    if any(m is None for m in configs):
        configs = _as_model_configs(model_configs)  # type: ignore[arg-type]
    else:
        configs = model_configs  # type: ignore[assignment]

    X = dataset.X
    y = dataset.y
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    cm_rows: list[dict[str, Any]] = []

    iterator = list(cv.split(X, y))
    for fold, (train_idx, test_idx) in enumerate(tqdm(iterator, desc="CV folds"), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for cfg in configs:  # type: ignore[union-attr]
            model_label = cfg.display_name or display_name(cfg.name)
            model = make_model(cfg.name, cfg.params)

            t0 = perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = perf_counter() - t0

            t1 = perf_counter()
            y_pred = model.predict(X_test)
            predict_seconds = perf_counter() - t1

            metrics = classification_metrics(y_test, y_pred)
            fold_rows.append(
                {
                    "fold": fold,
                    "method": model_label,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                    **metrics,
                }
            )

            for local_i, episode_index in enumerate(test_idx):
                pred_rows.append(
                    {
                        "fold": fold,
                        "method": model_label,
                        "episode_index": int(episode_index),
                        "episode_id": dataset.episode_ids[int(episode_index)],
                        "y_true": int(y_test[local_i]),
                        "y_true_name": dataset.class_names[int(y_test[local_i])],
                        "y_pred": int(y_pred[local_i]),
                        "y_pred_name": dataset.class_names[int(y_pred[local_i])],
                        "correct": bool(y_test[local_i] == y_pred[local_i]),
                    }
                )

            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(dataset.class_names)))
            for i, true_name in enumerate(dataset.class_names):
                for j, pred_name in enumerate(dataset.class_names):
                    cm_rows.append(
                        {
                            "fold": fold,
                            "method": model_label,
                            "true_label": i,
                            "pred_label": j,
                            "true_name": true_name,
                            "pred_name": pred_name,
                            "count": int(cm[i, j]),
                        }
                    )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(pred_rows)
    confusion = pd.DataFrame(cm_rows)
    summary = summarize_fold_metrics(fold_metrics)

    outputs = {
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "confusion_matrices": confusion,
        "summary": summary,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
        predictions.to_csv(out / "predictions.csv", index=False)
        confusion.to_csv(out / "confusion_matrices.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)

        metadata = pd.DataFrame(
            [
                {
                    "n_episodes": int(X.shape[0]),
                    "n_alarm_tags": int(X.shape[1]),
                    "n_time_steps": int(X.shape[2]),
                    "n_classes": int(len(dataset.class_names)),
                    "n_splits": int(n_splits),
                    "shuffle": bool(shuffle),
                    "random_state": int(random_state),
                }
            ]
        )
        metadata.to_csv(out / "metadata.csv", index=False)

        pd.DataFrame(
            {
                "class_id": np.arange(len(dataset.class_names), dtype=int),
                "class_name": dataset.class_names,
            }
        ).to_csv(out / "classes.csv", index=False)

        pd.DataFrame({"tag_id": np.arange(len(dataset.tag_names), dtype=int), "tag_name": dataset.tag_names}).to_csv(
            out / "alarm_tags.csv",
            index=False,
        )

    return outputs
