"""Train and evaluate the DCN-PLE-ESMM model family."""

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
from src.models.dcn_ple_esmm import DCNPLEESMM
from src.models.factory import build_multitask_model
from src.trainer.ctr_experiment import load_yaml, project_path
from src.trainer.dcn_ple_esmm_trainer import DCNPLEESMMTrainer
from src.trainer.multitask_experiment import run_multitask_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dcn_ple_esmm.yaml")
    parser.add_argument("--data-config")
    parser.add_argument("--experiment-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from")
    return parser.parse_args()


def build_dcn_ple_esmm(
    encoder: FeatureEncoder,
    model_config: Mapping[str, Any],
) -> DCNPLEESMM:
    model = build_multitask_model(encoder, model_config)
    if not isinstance(model, DCNPLEESMM):
        raise TypeError("model must belong to the DCN-PLE-ESMM family")
    return model


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(PROJECT_ROOT, args.config))
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("config must contain a model mapping")
    model_name = str(model_config.get("name", "")).lower()
    supported_models = ("dcn_ple_esmm", "target_aware_dcn_ple_esmm")
    if model_name not in supported_models:
        raise ValueError(
            "train_dcn_ple_esmm.py requires model.name in "
            f"{supported_models}; received {model_name!r}"
        )
    run_directory, result = run_multitask_experiment(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        expected_model_name=model_name,
        model_builder=build_dcn_ple_esmm,
        trainer_class=DCNPLEESMMTrainer,
        device_override=args.device,
        run_name=args.run_name,
        data_config_override=args.data_config,
        experiment_name_override=args.experiment_name,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        resume_from=args.resume_from,
    )
    report = {"valid": True, "run_directory": str(run_directory), **result}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
