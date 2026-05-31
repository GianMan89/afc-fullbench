"""Command-line interface for AFC-FullBench."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from afc_fullbench.data import load_alarm_series_dataset
from afc_fullbench.evaluation import run_cross_validation
from afc_fullbench.plotting import plot_results


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_from_config(config_path: str | Path) -> None:
    cfg = _load_yaml(config_path)

    data_cfg = cfg.get("data", {})
    cv_cfg = cfg.get("cv", {})
    output_dir = Path(cfg.get("output_dir", "results/experiment"))

    dataset = load_alarm_series_dataset(
        data_cfg.get("root", "data/tep"),
        max_time_steps=data_cfg.get("max_time_steps"),
    )

    run_cross_validation(
        dataset,
        model_configs=cfg.get("models", []),
        n_splits=int(cv_cfg.get("n_splits", 5)),
        shuffle=bool(cv_cfg.get("shuffle", True)),
        random_state=int(cv_cfg.get("random_state", 42)),
        output_dir=output_dir,
    )

    print(f"Wrote results to {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="afc-fullbench",
        description="Full-episode alarm flood classification benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a stratified k-fold experiment from YAML.")
    run_parser.add_argument("--config", required=True, help="Path to YAML configuration.")

    plot_parser = sub.add_parser("plot", help="Create standard plots from a result directory.")
    plot_parser.add_argument("--results-dir", required=True, help="Directory containing CSV outputs.")

    args = parser.parse_args(argv)

    if args.command == "run":
        run_from_config(args.config)
    elif args.command == "plot":
        paths = plot_results(args.results_dir)
        print("Created figures:")
        for path in paths:
            print(f"  {path}")
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
