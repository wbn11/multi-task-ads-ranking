# Ali-CCP raw data

Download Ali-CCP manually from the official Tianchi dataset page:

https://tianchi.aliyun.com/dataset/408

Do not commit the raw archives or extracted CSV files. The inspection command
accepts any of these common extraction layouts:

```text
data/raw/
├── sample_skeleton_train.csv
├── common_features_train.csv
├── sample_skeleton_test.csv
└── common_features_test.csv
```

```text
data/raw/
├── train/
│   ├── sample_skeleton_train.csv
│   └── common_features_train.csv
└── test/
    ├── sample_skeleton_test.csv
    └── common_features_test.csv
```

The archive may also use `sample_train/` and `sample_test/`; those directories
are detected automatically. Field-level meanings follow the official Ali-CCP
schema, while concrete categorical feature IDs remain anonymized.

After extraction, validate 100,000 training rows from the repository root:

```powershell
python -m src.data.inspect --raw-dir data/raw --split train --max-samples 100000
```

By default the command only writes the compact validation report:

- `data/processed/inspect_train_100000.json`: label, association, field-source,
  and cardinality statistics.

Expanded joined JSONL is optional because repeating long common feature lists can
produce a multi-GB file:

```powershell
python -m src.data.inspect --raw-dir data/raw --split train --max-samples 100000 `
  --output data/processed/joined_train_100k.jsonl
```
