"""Train the DCNv2 + PLE ablation with clicked-space masked CVR loss."""

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
from src.models.dcn_ple import DCNPLE
from src.models.factory import build_multitask_model
from src.trainer.cli import add_training_arguments, build_config_overrides
from src.trainer.multitask_experiment import run_multitask_experiment
from src.trainer.ple_trainer import PLETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(
        parser,
        default_config="configs/dcn_ple.yaml",
        model_fields=(
            "embedding_dim",
            "dropout",
            "gate_dropout",
            "num_cross_layers",
        ),
        loss_fields=("ctr_weight", "cvr_weight"),
    )
    return parser.parse_args()


def build_dcn_ple(
    encoder: FeatureEncoder,
    model_config: Mapping[str, Any],
) -> DCNPLE:
    model = build_multitask_model(encoder, model_config)
    if not isinstance(model, DCNPLE):
        raise TypeError("model must be DCNPLE")
    return model


def main() -> None:
    args = parse_args()
    run_directory, result = run_multitask_experiment(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        expected_model_name="dcn_ple",
        model_builder=build_dcn_ple,
        trainer_class=PLETrainer,
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
