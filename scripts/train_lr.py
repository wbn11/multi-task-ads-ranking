"""Train and evaluate the Ali-CCP LR CTR baseline."""

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
from src.trainer.cli import add_training_arguments, build_config_overrides
from src.trainer.ctr_experiment import run_ctr_experiment
from src.trainer.lr_trainer import LRTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, default_config="configs/lr.yaml")
    return parser.parse_args()


def build_lr(
    encoder: FeatureEncoder, model_config: Mapping[str, Any]
) -> LogisticRegressionCTR:
    del model_config
    return LogisticRegressionCTR.from_feature_encoder(encoder)


def main() -> None:
    args = parse_args()
    run_directory, result = run_ctr_experiment(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        expected_model_name="lr",
        model_builder=build_lr,
        trainer_class=LRTrainer,
        device_override=args.device,
        run_name=args.run_name,
        data_config_override=args.data_config,
        experiment_name_override=args.experiment_name,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        resume_from=args.resume_from,
        config_overrides=build_config_overrides(args),
    )
    report = {"valid": True, "run_directory": str(run_directory), **result}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
