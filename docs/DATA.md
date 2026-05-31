# Data layout

AFC-FullBench does not ship raw data. Place the datasets manually under:

```text
data/tep/
data/fcc/
```

Each dataset root must contain one subfolder per class. Each class folder must contain one CSV file per alarm-flood episode:

```text
data/tep/
├── class_01/
│   ├── run_0001.csv
│   └── run_0002.csv
├── class_02/
│   └── run_0001.csv
└── ...
```

Each CSV file is expected to contain a binary alarm activity matrix with one row per time step and one column per alarm tag. A time column named `Minutes`, `time`, `timestamp`, `t`, `minute`, `sec`, or `seconds` is removed automatically.

The loader converts each dataset to a tensor with shape:

```text
(n_episodes, n_alarm_tags, n_time_steps)
```

Episodes are truncated or zero-padded to the horizon specified in the YAML configuration.
