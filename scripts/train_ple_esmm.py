"""Train PLE with the ESMM entire-space CTR and CTCVR objective."""

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
from src.models.factory import build_multitask_model
from src.models.ple import PLE
from src.trainer.cli import add_training_arguments, build_config_overrides
from src.trainer.multitask_experiment import run_multitask_experiment
from src.trainer.ple_esmm_trainer import PLEESMMTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(
        parser,
        default_config="configs/ple_esmm.yaml",
        model_fields=("embedding_dim", "dropout", "gate_dropout"),
        loss_fields=("ctr_weight", "ctcvr_weight"),
    )
    return parser.parse_args()


def build_ple_esmm(
    encoder: FeatureEncoder,
    model_config: Mapping[str, Any],
) -> PLE:
    model = build_multitask_model(encoder, model_config)
    if not isinstance(model, PLE):
        raise TypeError("model must use the PLE architecture")
    return model


def main() -> None:
    args = parse_args()
    run_directory, result = run_multitask_experiment(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        expected_model_name="ple_esmm",
        model_builder=build_ple_esmm,
        trainer_class=PLEESMMTrainer,
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
