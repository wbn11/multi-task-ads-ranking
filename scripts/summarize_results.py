"""Build a compact, Git-friendly CSV from local experiment run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "run",
    "model",
    "seed",
    "dropout",
    "gate_dropout",
    "learning_rate",
    "negative_sampling_ratio",
    "auxiliary_cvr_weight",
    "parameter_count",
    "best_epoch",
    "best_validation_score",
    "total_seconds",
    "test_ctr_auc",
    "test_ctr_gauc",
    "test_ctr_log_loss",
    "test_cvr_auc",
    "test_cvr_gauc",
    "test_cvr_log_loss",
    "test_ctcvr_auc",
    "test_ctcvr_gauc",
    "test_ctcvr_log_loss",
    "platt_test_ctcvr_auc",
    "platt_test_ctcvr_log_loss",
    "platt_test_ctcvr_ece",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _metric(row: dict[str, Any], prefix: str, metrics: Any) -> None:
    if not isinstance(metrics, dict):
        return
    for key in ("auc", "gauc", "log_loss"):
        row[f"test_{prefix}_{key}"] = metrics.get(key)


def summarize_run(run_directory: Path) -> dict[str, Any]:
    metrics = _read_json(run_directory / "metrics.json")
    model = metrics.get("model_hyperparameters") or {}
    loss = metrics.get("loss_hyperparameters") or {}
    training = metrics.get("training") or {}
    sampling = metrics.get("negative_sampling") or {}
    auxiliary = loss.get("auxiliary_cvr_loss") or {}
    row: dict[str, Any] = {
        "run": run_directory.name,
        "model": metrics.get("model"),
        "seed": ((training.get("history") or [{}])[0].get("seed")),
        "dropout": model.get("dropout"),
        "gate_dropout": model.get("gate_dropout"),
        "learning_rate": (
            (training.get("history") or [{}])[0].get("learning_rate")
        ),
        "negative_sampling_ratio": sampling.get(
            "requested_negative_to_positive_ratio"
        ),
        "auxiliary_cvr_weight": (
            auxiliary.get("weight") if auxiliary.get("enabled") else None
        ),
        "parameter_count": metrics.get("parameter_count"),
        "best_epoch": training.get("best_epoch"),
        "best_validation_score": training.get(
            "best_validation_score", training.get("best_validation_auc")
        ),
        "total_seconds": training.get("total_seconds"),
    }

    # The seed lives in the resolved config snapshot rather than metrics in
    # older runs. Prefer that value when it is available.
    config_path = run_directory / "config.yaml"
    if config_path.is_file():
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            row["seed"] = config["model_config"]["training"]["seed"]
        except (KeyError, TypeError, ValueError):
            pass

    test = metrics.get("test") or {}
    if isinstance(test.get("ctr"), dict):
        for task in ("ctr", "cvr", "ctcvr"):
            _metric(row, task, test.get(task))
    else:
        _metric(row, "ctr", test)

    evaluation_path = run_directory / "evaluation_calibrated_multitask.json"
    if evaluation_path.is_file():
        evaluation = _read_json(evaluation_path)
        platt_ctcvr = (
            evaluation.get("learned_calibration", {})
            .get("platt", {})
            .get("test", {})
            .get("ctcvr", {})
        )
        if isinstance(platt_ctcvr, dict):
            row["platt_test_ctcvr_auc"] = platt_ctcvr.get("auc")
            row["platt_test_ctcvr_log_loss"] = platt_ctcvr.get("log_loss")
            row["platt_test_ctcvr_ece"] = (
                platt_ctcvr.get("calibration") or {}
            ).get("expected_calibration_error")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/experiment_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = [
        summarize_run(metrics_path.parent)
        for metrics_path in sorted(results_dir.glob("*/metrics.json"))
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {output_path}")


if __name__ == "__main__":
    main()
