# Experiment artifacts

`results/` is the local output root for training and evaluation. Raw run
directories can contain checkpoints close to 1 GB and are intentionally not
committed.

The repository keeps only `experiment_summary.csv`, a compact table generated
from every local `*/metrics.json` and optional calibration result:

```bash
python scripts/summarize_results.py
```

Every raw run remains reproducible because the runtime writes its fully
resolved `config.yaml` beside `metrics.json` and `train.log`. Keep important
checkpoints in external storage; do not use Git as model storage.
