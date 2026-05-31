# AFC-FullBench

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AFC-FullBench** is a lightweight benchmark repository for **full-episode alarm flood classification**. It provides the same baseline AFC method families and dataset layout used in AFC-RobustBench, but removes all online-prefix evaluation, perturbation, trace-repair, and robustness-analysis components.

The benchmark answers a simpler question:

> Given a complete extracted alarm-flood episode, how accurately can an AFC method classify the episode under stratified k-fold cross-validation?

All models are trained on complete clean training episodes and tested on complete held-out episodes.

---

## Scope

This repository is intentionally limited to offline/full-episode classification.

Implemented:

- loading binary alarm-series CSV files from class-folder datasets;
- converting complete alarm episodes to alarm sets, alarm activation sequences, and alarm series features;
- training AFC classifiers on complete episodes;
- stratified k-fold cross-validation;
- test-set prediction export;
- metric aggregation and summary tables;
- confusion matrices and summary accuracy plots.

Not included:

- online or prefix-based evaluation;
- perturbation or robustness testing;
- Monte-Carlo perturbation draws;
- trace repair;
- delayed-detection simulation;
- severity grids or robustness scores.

For robustness benchmarking, use `afc-robustbench`. For clean full-episode classification, use this repository.

---

## Evaluated AFC methods

| Abbreviation | Representation | Method family |
|---|---:|---|
| `WDI-1NN` | alarm set | weighted dissimilarity 1-nearest neighbor |
| `JAC-1NN` | alarm set | Jaccard distance 1-nearest neighbor |
| `EAC-1NN` | alarm sequence | exponentially attenuated components 1-nearest neighbor |
| `MBW-LR` | alarm sequence | modified bag-of-words with logistic regression |
| `ACM-SVM` | alarm series | alarm coactivation matrix with support vector machine |
| `CASIM` | alarm series | convolutional-kernel features with ridge classifier ensemble |

`CASIM` uses an optional `sktime` MultiRocket backend when available. If the optional backend is not installed, the package falls back to a deterministic random-convolution implementation for smoke tests and development.

---

## Repository layout

```text
.
├── configs/                    # YAML experiment configurations
├── data/                       # Local data directory; raw datasets are not tracked
│   ├── tep/
│   └── fcc/
├── docs/                       # Dataset layout and usage notes
├── scripts/                    # Utility scripts, including synthetic data generation
├── src/afc_fullbench/          # Installable Python package
│   ├── data.py                 # Dataset loading
│   ├── representations.py      # Set, sequence, and series features
│   ├── evaluation.py           # Stratified k-fold benchmark runner
│   ├── metrics.py              # Classification metrics
│   ├── plotting.py             # Result visualizations
│   ├── cli.py                  # Command-line interface
│   └── models/                 # AFC method implementations
└── tests/                      # Unit tests and smoke tests
```

---

## Installation

Create a fresh environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional CASIM/MultiRocket dependencies:

```bash
python -m pip install -e ".[dev,casim]"
```

The command-line entry point is:

```bash
afc-fullbench --help
```

---

## Data

Raw datasets are not included. Add them manually under `data/tep/` and `data/fcc/`.

The expected layout is one subfolder per class:

```text
data/tep/
├── class_01/
│   ├── run_0001.csv
│   └── run_0002.csv
├── class_02/
│   └── run_0001.csv
└── ...
```

Each CSV file should contain one row per time step and one column per alarm tag. A time column such as `Minutes`, `time`, `timestamp`, or `t` is detected and removed automatically.

Example:

```text
Minutes,XMEAS1_HI,XMEAS1_LO,XMEAS2_HI,...
1,0,0,0,...
2,0,0,0,...
3,0,1,0,...
```

The loader returns a binary tensor with shape:

```text
(n_episodes, n_alarm_tags, n_time_steps)
```

The paper datasets can be obtained from:

- Tennessee-Eastman Process alarm dataset: `https://dx.doi.org/10.21227/326k-qr90`
- Fluidized Catalytic Cracking alarm dataset: `https://doi.org/10.60517/2v23vv393`

---

## Quick smoke test

Generate a small synthetic dataset and run the benchmark:

```bash
python scripts/create_synthetic_dataset.py \
  --output data/smoke \
  --n-classes 3 \
  --n-runs-per-class 12

afc-fullbench run --config configs/smoke.yaml
afc-fullbench plot --results-dir results/smoke
```

The smoke test verifies the workflow and output generation. It is not intended as a scientific benchmark.

---

## Running TEP and FCC experiments

After placing the data under `data/tep/` and `data/fcc/`, run:

```bash
afc-fullbench run --config configs/tep.yaml
afc-fullbench run --config configs/fcc.yaml
```

Create standard plots from a result directory:

```bash
afc-fullbench plot --results-dir results/tep_full_episode
afc-fullbench plot --results-dir results/fcc_full_episode
```

---

## Output files

Each experiment writes CSV outputs to the configured result directory:

```text
results/<experiment>/
├── fold_metrics.csv            # one row per fold and method
├── predictions.csv             # one row per held-out episode prediction
├── confusion_matrices.csv       # fold-wise confusion counts
├── summary.csv                  # mean/std metrics by method
├── metadata.csv                 # dataset and CV metadata
├── classes.csv                  # class-id to class-name mapping
└── alarm_tags.csv               # alarm-tag index mapping
```

The standard metrics are:

- accuracy;
- balanced accuracy;
- macro F1;
- weighted F1;
- fit time and prediction time per fold.

---

## Configuration

Experiments are controlled by YAML files. A minimal configuration is:

```yaml
experiment_name: tep_full_episode
output_dir: results/tep_full_episode

data:
  root: data/tep
  max_time_steps: 60

cv:
  n_splits: 5
  shuffle: true
  random_state: 42

models:
  - name: WDI-1NN
  - name: JAC-1NN
  - name: EAC-1NN
    params:
      attenuation: 0.01
  - name: MBW-LR
    params:
      C: 1.0
      max_iter: 2000
  - name: ACM-SVM
  - name: CASIM
```

Model names are resolved through the model registry in `src/afc_fullbench/models/factory.py`.

---

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover dataset loading, representation extraction, and a small stratified cross-validation run.

---

## Citation

This repository is a companion artifact derived from the AFC-RobustBench code base and adapted to clean full-episode classification. Cite the corresponding AFC paper or project that uses this benchmark.

```bibtex
@misc{AFCFullBench2026,
  title  = {{AFC-FullBench}: Full-Episode Alarm Flood Classification Benchmark},
  author = {Manca, Gianluca and collaborators},
  year   = {2026},
  note   = {Software repository}
}
```

Please also cite the datasets when using them:

```bibtex
@misc{Manca2020_TEPAlarmDataset,
  author       = {Manca, Gianluca},
  title        = {{Tennessee-Eastman-Process} Alarm Management Dataset},
  howpublished = {IEEE Dataport},
  year         = {2020},
  doi          = {10.21227/326k-qr90}
}
```

```bibtex
@misc{Kunze2025_FCCAlarmDataset,
  author       = {Kunze, Franz C. and Manca, Gianluca and Fay, Alexander},
  title        = {{FCC} Alarm Dataset for Alarm Flood Classification},
  howpublished = {ReSeeD},
  year         = {2025},
  doi          = {10.60517/2v23vv393}
}
```

---

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
