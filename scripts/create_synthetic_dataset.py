#!/usr/bin/env python
"""Create a small synthetic alarm-flood dataset for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def create_episode(
    *,
    class_id: int,
    n_tags: int,
    n_time_steps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    X = np.zeros((n_time_steps, n_tags), dtype=int)

    # Class-specific diagnostic tags activate in class-specific windows.
    primary = [(class_id * 3 + offset) % n_tags for offset in range(3)]
    start = 3 + class_id * 4 + int(rng.integers(0, 3))
    duration = int(rng.integers(8, 18))
    for tag in primary:
        t0 = min(n_time_steps - 1, start + int(rng.integers(0, 5)))
        t1 = min(n_time_steps, t0 + duration + int(rng.integers(-2, 4)))
        X[t0:t1, tag] = 1

    # Propagated secondary alarms.
    for _ in range(4):
        tag = int(rng.integers(0, n_tags))
        t0 = int(rng.integers(start, max(start + 1, n_time_steps - 5)))
        t1 = min(n_time_steps, t0 + int(rng.integers(2, 8)))
        X[t0:t1, tag] = 1

    # Sparse nuisance alarms.
    noise = rng.random(size=X.shape) < 0.008
    X = np.maximum(X, noise.astype(int))

    df = pd.DataFrame(X, columns=[f"A{j:03d}" for j in range(n_tags)])
    df.insert(0, "Minutes", np.arange(n_time_steps))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/smoke", help="Output dataset directory.")
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--n-runs-per-class", type=int, default=12)
    parser.add_argument("--n-tags", type=int, default=24)
    parser.add_argument("--n-time-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    rng = np.random.default_rng(args.seed)

    for c in range(args.n_classes):
        folder = out / f"class_{c + 1:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for r in range(args.n_runs_per_class):
            df = create_episode(
                class_id=c,
                n_tags=args.n_tags,
                n_time_steps=args.n_time_steps,
                rng=rng,
            )
            df.to_csv(folder / f"run_{r + 1:04d}.csv", index=False)

    print(f"Wrote synthetic dataset to {out}")


if __name__ == "__main__":
    main()
