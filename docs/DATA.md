# Data layout

AFC-FullBench does not track raw datasets. Place binary alarm activity CSV files under `data/tep/` and `data/fcc/` with one subfolder per class.

```text
data/tep/
├── class_01/
│   ├── run_0001.csv
│   └── run_0002.csv
├── class_02/
│   └── run_0001.csv
└── ...
```

Each CSV file should contain one row per time step and one column per alarm tag. Optional time columns named `Minutes`, `time`, `timestamp`, `t`, `minute`, `sec`, or `seconds` are ignored by the loader.

The paper datasets can be obtained from:

- Tennessee-Eastman Process alarm dataset: https://dx.doi.org/10.21227/326k-qr90
- Fluidized Catalytic Cracking alarm dataset: https://doi.org/10.60517/2v23vv393
