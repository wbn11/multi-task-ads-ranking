"""Evaluate and calibrate a saved LR CTR checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_encoder import FeatureEncoder
from src.models.lr import LogisticRegressionCTR
from src.trainer.ctr_evaluation import evaluate_ctr_run
from src.trainer.lr_trainer import LRTrainer


def build_lr(
    encoder: FeatureEncoder,
    model_config: Mapping[str, Any],
) -> LogisticRegressionCTR:
    del model_config
    return LogisticRegressionCTR.from_feature_encoder(encoder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument(
        "--processed-dir",
        help="Override the processed dataset path saved in the run config.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of probability bins used by ECE and curve data.",
    )
    parser.add_argument(
        "--calibration-binning",
        choices=("equal_frequency", "equal_width"),
        default="equal_frequency",
        help="Probability binning strategy for calibration evaluation.",
    )
    parser.add_argument(
        "--platt-l2",
        type=float,
        default=1e-6,
        help="L2 regularization applied to the Platt slope.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, result = evaluate_ctr_run(
        project_root=PROJECT_ROOT,
        run_directory=args.run_directory,
        expected_model_name="lr",
        model_builder=build_lr,
        trainer_class=LRTrainer,
        device_override=args.device,
        processed_dir_override=args.processed_dir,
        calibration_num_bins=args.calibration_bins,
        calibration_binning_strategy=args.calibration_binning,
        platt_l2_regularization=args.platt_l2,
    )
    report = {"valid": True, "output_path": str(output_path), **result}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
