from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from afc_fullbench.data import load_alarm_series_dataset
from afc_fullbench.evaluation import run_cross_validation
from afc_fullbench.models import ACMSVM, CASIM, EAC1NN, Jaccard1NN, MBWLogisticRegression, WDI1NN
from afc_fullbench.representations import activation_sequence_from_series, active_set_vector


def _write_tiny_dataset(root: Path, *, n_classes: int = 3, n_runs: int = 6) -> None:
    """Create a tiny class-folder alarm dataset for tests."""
    rng = np.random.default_rng(123)
    for cls in range(n_classes):
        folder = root / f"class_{cls + 1:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for run in range(n_runs):
            X = np.zeros((24, 5), dtype=int)
            # Class-specific main alarm plus a little random background activity.
            X[2 + cls : 8 + cls, cls] = 1
            X[10 + cls : 14 + cls, (cls + 1) % 5] = 1
            noise_tag = int(rng.integers(0, 5))
            X[int(rng.integers(0, 24)), noise_tag] = 1
            df = pd.DataFrame(X, columns=[f"A{i}" for i in range(5)])
            df.insert(0, "Minutes", np.arange(len(df)))
            df.to_csv(folder / f"run_{run:03d}.csv", index=False)


def test_load_dataset_and_representations(tmp_path: Path):
    root = tmp_path / "data"
    _write_tiny_dataset(root)
    dataset = load_alarm_series_dataset(root, max_time_steps=24)

    assert dataset.X.shape == (18, 5, 24)
    assert dataset.y.shape == (18,)
    assert len(dataset.class_names) == 3
    assert active_set_vector(dataset.X[0]).shape == (5,)
    assert len(activation_sequence_from_series(dataset.X[0])) >= 2


def test_models_fit_predict_on_complete_episodes(tmp_path: Path):
    root = tmp_path / "data"
    _write_tiny_dataset(root)
    dataset = load_alarm_series_dataset(root, max_time_steps=24)
    X, y = dataset.X, dataset.y

    models = [
        WDI1NN(template_threshold=0.5),
        Jaccard1NN(),
        EAC1NN(attenuation=0.01),
        MBWLogisticRegression(max_iter=300),
        ACMSVM(probability=True),
        CASIM(num_features=32, backend="lite", random_state=1),
    ]

    for model in models:
        model.fit(X, y)
        pred = model.predict(X[:4])
        assert pred.shape == (4,)


def test_repeated_cross_validation_outputs(tmp_path: Path):
    root = tmp_path / "data"
    _write_tiny_dataset(root)
    dataset = load_alarm_series_dataset(root, max_time_steps=24)
    out = tmp_path / "results"

    outputs = run_cross_validation(
        dataset,
        model_configs=[
            {"name": "WDI-1NN"},
            {"name": "JAC-1NN"},
            {"name": "EAC-1NN", "params": {"attenuation": 0.01}},
        ],
        n_splits=3,
        n_repeats=2,
        shuffle=True,
        random_state=7,
        n_jobs=1,
        output_dir=out,
    )

    assert outputs["fold_metrics"].shape[0] == 3 * 2 * 3
    assert set(outputs["fold_metrics"]["repeat"]) == {1, 2}
    assert (out / "summary.csv").exists()
    assert (out / "predictions.csv").exists()
