"""Plotting utilities for AFC-FullBench outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _safe_label(value: str) -> str:
    """Return a filesystem-safe lowercase label."""

    safe = str(value).strip().lower().replace(" ", "_").replace("/", "_")
    for char in ["+", ":", ";", "(", ")", "[", "]"]:
        safe = safe.replace(char, "")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def plot_summary_bar(
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_label: str | None = None,
    metric: str = "accuracy",
) -> Path:
    """Plot mean cross-validation performance by method.

    Parameters
    ----------
    summary:
        Summary table created by :func:`afc_fullbench.evaluation.run_cross_validation`.
    output_dir:
        Folder in which the PDF and PNG files are written.
    dataset_label:
        Optional label used in the figure title and filename.
    metric:
        Metric prefix, for example ``"accuracy"`` or ``"macro_f1"``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary.columns:
        raise ValueError(f"summary must contain `{mean_col}`.")

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(summary))
    y = summary[mean_col].astype(float).to_numpy()
    ax.bar(x, y)
    if std_col in summary.columns:
        ax.errorbar(
            x,
            y,
            yerr=summary[std_col].astype(float).to_numpy(),
            fmt="none",
            capsize=3,
            linewidth=1.0,
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("AFC method")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"].astype(str), rotation=30, ha="right")
    if dataset_label:
        ax.set_title(f"{dataset_label}: full-episode classification")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()

    prefix = f"{_safe_label(dataset_label)}_" if dataset_label else ""
    out = output_dir / f"{prefix}{metric}_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    classes: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_label: str | None = None,
    normalize: str | None = "true",
) -> list[Path]:
    """Plot one pooled confusion matrix per method.

    Predictions from all repeated-CV fold units are pooled before computing the
    confusion matrix.  This produces one compact diagnostic per method and
    dataset.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = classes["class_id"].astype(int).to_numpy()
    class_names = classes["class_name"].astype(str).tolist()

    paths: list[Path] = []
    for method, group in predictions.groupby("method", sort=False):
        y_true = group["y_true"].astype(int).to_numpy()
        y_pred = group["y_pred"].astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=class_ids, normalize=normalize)

        fig, ax = plt.subplots(figsize=(4.9, 4.3))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, values_format=".2f" if normalize else "d", colorbar=True)
        title = str(method) if dataset_label is None else f"{dataset_label}: {method}"
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        dataset_prefix = f"{_safe_label(dataset_label)}_" if dataset_label else ""
        out = output_dir / f"{dataset_prefix}confusion_{_safe_label(method)}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    return paths


def plot_results(results_dir: str | Path, *, figures_dir: str | Path | None = None) -> list[Path]:
    """Create standard plots from a result directory."""

    results_dir = Path(results_dir)
    output_dir = Path(figures_dir) if figures_dir is not None else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "summary.csv")
    predictions = pd.read_csv(results_dir / "predictions.csv")
    classes = pd.read_csv(results_dir / "classes.csv")

    dataset_label = results_dir.name.replace("_full_episode", "").upper()
    paths = [plot_summary_bar(summary, output_dir, dataset_label=dataset_label, metric="accuracy")]
    paths.extend(
        plot_confusion_matrices(
            predictions,
            classes,
            output_dir,
            dataset_label=dataset_label,
            normalize="true",
        )
    )
    return paths
