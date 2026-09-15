"""Train and evaluate ESMM on the processed Ali-CCP splits."""

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
from src.models.esmm import ESMM
from src.trainer.esmm_trainer import ESMMTrainer
from src.trainer.multitask_experiment import run_multitask_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/esmm.yaml")
    parser.add_argument("--data-config")
    parser.add_argument("--experiment-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from")
    return parser.parse_args()


def build_esmm(
    encoder: FeatureEncoder,
    model_config: Mapping[str, Any],
) -> ESMM:
    return ESMM.from_feature_encoder(
        encoder,
        embedding_dim=int(model_config["embedding_dim"]),
        shared_hidden_dims=tuple(
            int(value) for value in model_config["shared_hidden_dims"]
        ),
        tower_hidden_dims=tuple(
            int(value) for value in model_config["tower_hidden_dims"]
        ),
        dropout=float(model_config["dropout"]),
        embedding_pooling=str(model_config["embedding_pooling"]),
    )


def main() -> None:
    args = parse_args()
    run_directory, result = run_multitask_experiment(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        expected_model_name="esmm",
        model_builder=build_esmm,
        trainer_class=ESMMTrainer,
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
