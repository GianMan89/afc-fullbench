# AFC-FullBench

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AFC-FullBench** is a compact benchmark repository for **full-episode alarm flood classification**. It reuses the baseline AFC method families and dataset representation conventions of AFC-RobustBench, but removes all online-prefix evaluation, perturbations, delayed-detection simulation, trace repair, and robustness aggregation.

The benchmark answers one focused question:

> Given a complete extracted alarm-flood episode, how accurately can an AFC method classify the episode under repeated stratified cross-validation?

All classifiers are trained on complete clean training episodes and tested on complete held-out episodes.

---

## Scope

Implemented:

- loading binary alarm-series CSV files from class-folder datasets;
- deriving full-episode alarm-set, alarm-sequence, and alarm-series representations;
- training AFC classifiers on complete episodes only;
- repeated stratified k-fold cross-validation;
- parallel evaluation across model/split tasks using `joblib`;
- prediction export, metric aggregation, and summary tables;
- confusion matrices and summary accuracy plots;
- one Jupyter notebook for TEP and FCC training, testing, and visualization.

Not included:

- online or prefix-based evaluation;
- perturbations or robustness testing;
- Monte-Carlo perturbation draws;
- delayed-detection simulation;
- severity grids or robustness scores;
- synthetic smoke-data generation scripts.

Use AFC-RobustBench for perturbation-based robustness benchmarking. Use AFC-FullBench for clean full-episode classification.

---

## Evaluated AFC methods

| Abbreviation | Representation | Method family |
|---|---:|---|
| `WDI-1NN` | alarm set | weighted dissimilarity template classifier |
| `JAC-1NN` | alarm set | Jaccard distance 1-nearest neighbor |
| `EAC-1NN` | alarm sequence | exponentially attenuated components 1-nearest neighbor |
| `MBW-LR` | alarm sequence | modified bag-of-words with logistic regression |
| `ACM-SVM` | alarm series | alarm coactivation matrix with support vector machine |
| `CASIM` | alarm series | convolutional-kernel features with ridge classifier |

The implementations follow the corresponding AFC-RobustBench baseline definitions, but are used only in their full-episode/offline form. Online-prefix wrapper parameters from AFC-RobustBench configurations are ignored automatically if present.

`CASIM` supports an optional `sktime` MultiRocket backend. If the optional backend is unavailable or incompatible, `backend: auto` falls back to a deterministic CASIM-lite random-convolution implementation. The fallback is suitable for development and reproducibility checks; final experiments should use a consistent dependency environment.

---

## Repository layout

```text
.
├── configs/                    # TEP and FCC YAML experiment configurations
├── data/                       # Local data directory; raw datasets are not tracked
│   ├── tep/
│   └── fcc/
├── docs/                       # Dataset layout and usage notes
├── figures/                    # Generated notebook/CLI figures; ignored by git
├── notebooks/                  # Full-episode training/testing/visualization notebook
├── results/                    # Generated CSV outputs; ignored by git
├── src/afc_fullbench/          # Installable Python package
│   ├── data.py                 # Dataset loading
│   ├── representations.py      # Set, sequence, and series feature extraction
│   ├── evaluation.py           # Repeated stratified CV benchmark runner
│   ├── metrics.py              # Classification metrics
│   ├── plotting.py             # Result visualizations
│   ├── cli.py                  # Command-line interface
│   └── models/                 # AFC method implementations
└── tests/                      # Unit tests using temporary synthetic data
```

Raw data, result CSV files, and generated figures are intentionally excluded from version control.

---

## Installation

Create a fresh Python environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
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

The datasets used in our AFC work can be obtained from:

- Tennessee-Eastman Process alarm dataset: `https://dx.doi.org/10.21227/326k-qr90`
- Fluidized Catalytic Cracking alarm dataset: `https://doi.org/10.60517/2v23vv393`

---

## Running TEP and FCC experiments from the CLI

After placing the data under `data/tep/` and `data/fcc/`, run:

```bash
afc-fullbench run --config configs/tep.yaml
afc-fullbench run --config configs/fcc.yaml
```

Create standard plots from a result directory:

```bash
afc-fullbench plot --results-dir results/tep_full_episode --figures-dir figures
afc-fullbench plot --results-dir results/fcc_full_episode --figures-dir figures
```

---

## Notebook workflow

The main interactive workflow is:

```text
notebooks/01_full_episode_training_testing_visualization.ipynb
```

The notebook:

1. loads the TEP and FCC configurations;
2. runs repeated stratified cross-validation for each dataset;
3. trains and tests all configured full-episode AFC methods;
4. writes result CSV files to `results/`;
5. writes all generated figures directly to `figures/`;
6. creates a combined summary table for both datasets.

Set `RUN_CV = False` in the notebook to regenerate tables and figures from existing CSV outputs without rerunning the classifiers.

---

## Configuration

Experiments are controlled by YAML files. The important fields are:

```yaml
parallel:
  n_jobs: -1
  backend: loky
  inner_max_num_threads: 1

cv:
  n_splits: 5
  n_repeats: 5
  shuffle: true
  random_state: 42

models:
  - name: wdi_1nn
    params:
      template_threshold: 0.5
  - name: eac_1nn
    params:
      attenuation: 0.01
      time_scale: event_index
      distance: euclidean
      normalize: true
  - name: casim
    params:
      num_features: 672
      n_estimators: 1
      backend: auto
```

The evaluation is parallelized over the Cartesian product of repeated-CV splits and model configurations. To run serially, set `parallel.n_jobs: 1`.

---

## Output files

Each experiment writes CSV outputs to the configured result directory:

```text
results/<experiment>/
├── fold_metrics.csv            # one row per repeat/fold/method
├── predictions.csv             # one row per held-out episode prediction
├── confusion_matrices.csv       # repeat/fold-wise confusion counts
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
- fit time and prediction time per fold-repeat unit.

---

## Testing

Run the test suite with:

```bash
pytest
```

The tests generate temporary synthetic class-folder data and do not require the TEP or FCC datasets.

---

## Citation

This repository is a companion artifact derived from the AFC-RobustBench code base and adapted to clean full-episode alarm flood classification.

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
