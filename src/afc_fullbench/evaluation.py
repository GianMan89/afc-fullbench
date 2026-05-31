"""Repeated stratified cross-validation for full-episode AFC classifiers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from tqdm.auto import tqdm

from afc_fullbench.data import AlarmDataset
from afc_fullbench.metrics import classification_metrics
from afc_fullbench.models.factory import display_name, make_model


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one AFC model family.

    Parameters
    ----------
    name:
        Registry key, for example ``"EAC-1NN"`` or ``"casim"``.
    params:
        Hyperparameters passed to the model constructor.
    display_name:
        Optional label used in result tables and figures.
    """

    name: str
    params: dict[str, Any]
    display_name: str | None = None


def _first_params_from_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return constructor parameters from either ``params`` or ``params_grid``.

    AFC-RobustBench configurations use ``params_grid`` because they support
    validation-based model selection.  AFC-FullBench intentionally performs a
    fixed-parameter benchmark, so the first grid entry is used when a grid is
    supplied.
    """

    if "params" in item:
        return dict(item.get("params") or {})
    grid = item.get("params_grid")
    if grid is None:
        return {}
    if not isinstance(grid, list) or not grid:
        raise ValueError(f"params_grid must be a non-empty list if supplied: {item}")
    return dict(grid[0] or {})


def _as_model_configs(configs: list[dict[str, Any]] | list[ModelConfig]) -> list[ModelConfig]:
    """Normalize dictionaries or ``ModelConfig`` objects to a typed list."""

    out: list[ModelConfig] = []
    for item in configs:
        if isinstance(item, ModelConfig):
            out.append(item)
            continue
        if "name" not in item:
            raise ValueError(f"Model entry is missing `name`: {item}")
        name = str(item["name"])
        params = _first_params_from_item(item)
        out.append(
            ModelConfig(
                name=name,
                params=params,
                display_name=item.get("display_name", display_name(name)),
            )
        )
    return out


def _split_indices(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    shuffle: bool,
    random_state: int,
) -> list[tuple[int, int, int, np.ndarray, np.ndarray]]:
    """Create repeated stratified train/test splits.

    Returns tuples ``(split_id, repeat, fold, train_idx, test_idx)``.  For
    ``n_repeats == 1``, ordinary ``StratifiedKFold`` is used so that
    ``shuffle=False`` remains meaningful.  For more than one repeat,
    ``RepeatedStratifiedKFold`` is used.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    min_class_count = int(np.min(np.bincount(np.asarray(y, dtype=int))))
    if n_splits > min_class_count:
        raise ValueError(
            f"n_splits={n_splits} exceeds the smallest class size ({min_class_count}). "
            "Reduce n_splits or add more episodes per class."
        )

    splits: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    if n_repeats == 1:
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            splits.append((fold - 1, 1, fold, train_idx, test_idx))
        return splits

    if not shuffle:
        raise ValueError("Repeated stratified CV requires shuffle=True.")
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    for split_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        repeat = split_id // n_splits + 1
        fold = split_id % n_splits + 1
        splits.append((split_id, repeat, fold, train_idx, test_idx))
    return splits


def _evaluate_one_model_split(
    *,
    X: np.ndarray,
    y: np.ndarray,
    episode_ids: list[str],
    class_names: list[str],
    split_id: int,
    repeat: int,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ModelConfig,
) -> dict[str, Any]:
    """Fit and test one model on one repeated-CV split.

    This function is intentionally top-level and side-effect-free so it can be
    executed by joblib worker processes.
    """

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model_label = cfg.display_name or display_name(cfg.name)

    model = make_model(cfg.name, cfg.params)

    t0 = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - t0

    t1 = perf_counter()
    y_pred = np.asarray(model.predict(X_test))
    predict_seconds = perf_counter() - t1

    metrics = classification_metrics(y_test, y_pred)
    fold_row = {
        "split_id": int(split_id),
        "repeat": int(repeat),
        "fold": int(fold),
        "method": model_label,
        "model_key": cfg.name,
        "model_params": json.dumps(cfg.params, sort_keys=True),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        **metrics,
    }

    pred_rows = []
    for local_i, episode_index in enumerate(test_idx):
        pred_rows.append(
            {
                "split_id": int(split_id),
                "repeat": int(repeat),
                "fold": int(fold),
                "method": model_label,
                "episode_index": int(episode_index),
                "episode_id": episode_ids[int(episode_index)],
                "y_true": int(y_test[local_i]),
                "y_true_name": class_names[int(y_test[local_i])],
                "y_pred": int(y_pred[local_i]),
                "y_pred_name": class_names[int(y_pred[local_i])],
                "correct": bool(y_test[local_i] == y_pred[local_i]),
            }
        )

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    cm_rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            cm_rows.append(
                {
                    "split_id": int(split_id),
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "method": model_label,
                    "true_label": i,
                    "pred_label": j,
                    "true_name": true_name,
                    "pred_name": pred_name,
                    "count": int(cm[i, j]),
                }
            )

    return {"fold_row": fold_row, "pred_rows": pred_rows, "cm_rows": cm_rows}


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return mean/std metric summary by method over fold-repeat units."""

    excluded = {"split_id", "repeat", "fold", "method", "model_key", "model_params", "train_size", "test_size"}
    metric_cols = [
        c
        for c in fold_metrics.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(fold_metrics[c])
    ]
    rows = []
    for method, group in fold_metrics.groupby("method", sort=False):
        row: dict[str, Any] = {"method": method, "n_units": int(len(group))}
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
    n_repeats: int = 1,
    shuffle: bool = True,
    random_state: int = 42,
    n_jobs: int = 1,
    backend: str = "loky",
    inner_max_num_threads: int | None = 1,
    pre_dispatch: str | int = "2*n_jobs",
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run repeated stratified full-episode classification.

    All models are trained on complete clean training episodes and tested on
    complete held-out episodes.  No online prefixes, perturbations, Monte-Carlo
    draws, trace repair, or robustness scores are used.

    Parallelization is performed over the Cartesian product of repeated-CV
    splits and model configurations.  The returned data frames are deterministic
    in row order regardless of ``n_jobs``.
    """

    if not model_configs:
        raise ValueError("At least one model configuration is required.")

    configs = _as_model_configs(model_configs)
    X = np.asarray(dataset.X)
    y = np.asarray(dataset.y)

    splits = _split_indices(
        X,
        y,
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        shuffle=bool(shuffle),
        random_state=int(random_state),
    )

    tasks = []
    for split_id, repeat, fold, train_idx, test_idx in splits:
        for model_order, cfg in enumerate(configs):
            tasks.append((split_id, repeat, fold, train_idx, test_idx, model_order, cfg))

    def run_task(task):
        split_id, repeat, fold, train_idx, test_idx, model_order, cfg = task
        result = _evaluate_one_model_split(
            X=X,
            y=y,
            episode_ids=dataset.episode_ids,
            class_names=dataset.class_names,
            split_id=split_id,
            repeat=repeat,
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            cfg=cfg,
        )
        result["model_order"] = model_order
        return result

    if int(n_jobs) == 1:
        results = [run_task(task) for task in tqdm(tasks, desc="CV model tasks")]
    else:
        config_kwargs = {"backend": backend}
        if inner_max_num_threads is not None:
            config_kwargs["inner_max_num_threads"] = inner_max_num_threads
        with parallel_config(**config_kwargs):
            results = Parallel(n_jobs=int(n_jobs), pre_dispatch=pre_dispatch)(
                delayed(run_task)(task) for task in tqdm(tasks, desc="Submitting CV model tasks")
            )

    # Sort explicitly by split and model order to make outputs independent of job completion order.
    results = sorted(
        results,
        key=lambda r: (
            int(r["fold_row"]["split_id"]),
            int(r["model_order"]),
        ),
    )

    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    cm_rows: list[dict[str, Any]] = []
    for result in results:
        fold_rows.append(result["fold_row"])
        pred_rows.extend(result["pred_rows"])
        cm_rows.extend(result["cm_rows"])

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
                    "n_repeats": int(n_repeats),
                    "shuffle": bool(shuffle),
                    "random_state": int(random_state),
                    "n_jobs": int(n_jobs),
                    "parallel_backend": backend,
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

        pd.DataFrame(
            {
                "tag_id": np.arange(len(dataset.tag_names), dtype=int),
                "tag_name": dataset.tag_names,
            }
        ).to_csv(out / "alarm_tags.csv", index=False)

    return outputs
