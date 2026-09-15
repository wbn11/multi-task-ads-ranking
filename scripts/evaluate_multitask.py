"""Evaluate and calibrate a saved multi-task CTR/CVR/CTCVR checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.multitask_evaluation import evaluate_multitask_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--processed-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--calibration-binning",
        choices=("equal_frequency", "equal_width"),
        default="equal_frequency",
    )
    parser.add_argument("--platt-l2", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, result = evaluate_multitask_run(
        project_root=PROJECT_ROOT,
        run_directory=args.run_directory,
        device_override=args.device,
        processed_dir_override=args.processed_dir,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        calibration_num_bins=args.calibration_bins,
        calibration_binning_strategy=args.calibration_binning,
        platt_l2_regularization=args.platt_l2,
    )
    report = {"valid": True, "output_path": str(output_path), **result}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
