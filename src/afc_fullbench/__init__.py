"""AFC-FullBench: full-episode alarm flood classification benchmark."""

from afc_fullbench.data import AlarmDataset, load_alarm_series_dataset
from afc_fullbench.evaluation import run_cross_validation

__all__ = [
    "AlarmDataset",
    "load_alarm_series_dataset",
    "run_cross_validation",
]

__version__ = "0.1.0"
