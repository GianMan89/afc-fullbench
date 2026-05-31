"""Plotting utilities for AFC-FullBench outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_summary_bar(summary: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot mean accuracy by method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "accuracy_mean" not in summary.columns:
        raise ValueError("summary.csv must contain `accuracy_mean`.")

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.bar(summary["method"].astype(str), summary["accuracy_mean"].astype(float))
    if "accuracy_std" in summary.columns:
        ax.errorbar(
            np.arange(len(summary)),
            summary["accuracy_mean"].astype(float),
            yerr=summary["accuracy_std"].astype(float),
            fmt="none",
            capsize=3,
            linewidth=1.0,
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("AFC method")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()

    out = output_dir / "summary_accuracy.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    classes: pd.DataFrame,
    output_dir: str | Path,
    *,
    normalize: str | None = "true",
) -> list[Path]:
    """Plot one pooled confusion matrix per method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = classes["class_id"].astype(int).to_numpy()
    class_names = classes["class_name"].astype(str).tolist()

    paths: list[Path] = []
    for method, group in predictions.groupby("method", sort=False):
        y_true = group["y_true"].astype(int).to_numpy()
        y_pred = group["y_pred"].astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=class_ids, normalize=normalize)

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, values_format=".2f" if normalize else "d", colorbar=True)
        ax.set_title(str(method))
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        safe_method = str(method).lower().replace(" ", "_").replace("/", "_")
        out = output_dir / f"confusion_{safe_method}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    return paths


def plot_results(results_dir: str | Path) -> list[Path]:
    """Create standard plots from a result directory."""
    results_dir = Path(results_dir)
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "summary.csv")
    predictions = pd.read_csv(results_dir / "predictions.csv")
    classes = pd.read_csv(results_dir / "classes.csv")

    paths = [plot_summary_bar(summary, output_dir)]
    paths.extend(plot_confusion_matrices(predictions, classes, output_dir))
    return paths
