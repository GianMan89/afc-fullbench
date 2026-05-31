from pathlib import Path

import numpy as np
import pandas as pd

from afc_fullbench.data import load_alarm_series_dataset
from afc_fullbench.evaluation import run_cross_validation
from afc_fullbench.representations import activation_events, alarm_set_features


def _write_small_dataset(root: Path) -> None:
    for class_id in range(2):
        folder = root / f"class_{class_id + 1:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for run in range(4):
            arr = np.zeros((20, 6), dtype=int)
            if class_id == 0:
                arr[2:10, 0] = 1
                arr[5:12, 1] = 1
            else:
                arr[3:11, 3] = 1
                arr[8:15, 4] = 1
            df = pd.DataFrame(arr, columns=[f"A{i}" for i in range(6)])
            df.insert(0, "Minutes", np.arange(20))
            df.to_csv(folder / f"run_{run}.csv", index=False)


def test_loading_and_representations(tmp_path):
    _write_small_dataset(tmp_path)
    ds = load_alarm_series_dataset(tmp_path)
    assert ds.X.shape == (8, 6, 20)
    assert len(ds.class_names) == 2
    assert activation_events(ds.X[0])
    assert alarm_set_features(ds.X).shape == (8, 6)


def test_cross_validation_smoke(tmp_path):
    _write_small_dataset(tmp_path / "data")
    ds = load_alarm_series_dataset(tmp_path / "data")
    outputs = run_cross_validation(
        ds,
        model_configs=[{"name": "WDI-1NN"}, {"name": "JAC-1NN"}, {"name": "MBW-LR"}],
        n_splits=2,
        output_dir=tmp_path / "results",
    )
    assert set(outputs) == {"fold_metrics", "predictions", "confusion_matrices", "summary"}
    assert (tmp_path / "results" / "summary.csv").exists()
    assert outputs["summary"]["accuracy_mean"].min() >= 0.0
