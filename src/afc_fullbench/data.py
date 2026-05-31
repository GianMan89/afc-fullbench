"""Dataset loading utilities for full-episode alarm flood classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_DEFAULT_TIME_COLUMNS = {"time", "timestamp", "t", "minutes", "minute", "sec", "seconds"}


@dataclass(frozen=True)
class AlarmDataset:
    """Container for a full-episode alarm classification dataset.

    Attributes
    ----------
    X:
        Binary alarm activity tensor with shape ``(n_episodes, n_tags, n_time_steps)``.
    y:
        Integer class labels with shape ``(n_episodes,)``.
    class_names:
        Class-folder names sorted lexicographically.
    tag_names:
        Alarm tag names corresponding to the second axis of ``X``.
    episode_ids:
        Relative CSV paths, one per episode.
    """

    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    tag_names: list[str]
    episode_ids: list[str]


def _is_time_column(column: str, time_columns: Iterable[str]) -> bool:
    normalized = str(column).strip().lower()
    return normalized in {c.lower() for c in time_columns}


def _read_alarm_csv(path: Path, *, time_columns: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [c for c in df.columns if not _is_time_column(c, time_columns)]
    if not keep:
        raise ValueError(f"No alarm columns found in {path}.")
    out = df[keep].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _discover_class_folders(root: Path) -> list[Path]:
    folders = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not folders:
        raise ValueError(
            f"Expected one subfolder per class under {root}. "
            "For example: data/tep/class_01/run_0001.csv."
        )
    return folders


def _csv_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.csv") if p.is_file())


def load_alarm_series_dataset(
    root: str | Path,
    *,
    max_time_steps: int | None = None,
    time_columns: Iterable[str] = _DEFAULT_TIME_COLUMNS,
    dtype: type | str = np.float32,
) -> AlarmDataset:
    """Load a class-folder alarm-series dataset.

    Parameters
    ----------
    root:
        Dataset root containing one subfolder per class.
    max_time_steps:
        Optional fixed time horizon. Episodes longer than this value are truncated;
        shorter episodes are padded with zeros. If omitted, the maximum episode
        length in the dataset is used.
    time_columns:
        Column names interpreted as time columns and excluded from the alarm matrix.
    dtype:
        Numpy dtype of the returned tensor.
    """

    root = Path(root)
    dtype = np.dtype(dtype)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    class_folders = _discover_class_folders(root)
    class_names = [p.name for p in class_folders]

    records: list[tuple[Path, int]] = []
    all_tags: set[str] = set()
    max_len = 0

    for label, folder in enumerate(class_folders):
        files = _csv_files(folder)
        if not files:
            raise ValueError(f"No CSV files found in class folder {folder}.")
        for file in files:
            df = _read_alarm_csv(file, time_columns=time_columns)
            all_tags.update(map(str, df.columns))
            max_len = max(max_len, len(df))
            records.append((file, label))

    tag_names = sorted(all_tags)
    horizon = int(max_time_steps) if max_time_steps is not None else max_len
    if horizon <= 0:
        raise ValueError("max_time_steps must be positive.")

    X = np.zeros((len(records), len(tag_names), horizon), dtype=dtype)
    y = np.zeros(len(records), dtype=np.int64)
    episode_ids: list[str] = []

    for idx, (file, label) in enumerate(records):
        df = _read_alarm_csv(file, time_columns=time_columns)
        df = df.reindex(columns=tag_names, fill_value=0.0)
        arr = df.to_numpy(dtype=dtype, copy=True).T  # tags x time
        length = min(arr.shape[1], horizon)
        X[idx, :, :length] = arr[:, :length]
        y[idx] = label
        episode_ids.append(str(file.relative_to(root)))

    # Force binary semantics. Any positive value is interpreted as active.
    X = (X > 0).astype(dtype, copy=False)

    return AlarmDataset(
        X=X,
        y=y,
        class_names=class_names,
        tag_names=tag_names,
        episode_ids=episode_ids,
    )
