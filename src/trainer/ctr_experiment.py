"""Shared orchestration for single-task CTR experiments."""

from __future__ import annotations

import json
import logging
import random
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import yaml
from torch import nn

from src.data.dataloader import create_dataloader
from src.data.dataset import AdsDataset
from src.data.feature_encoder import FeatureEncoder
from src.data.sampler import configure_negative_downsampling
from src.trainer.ctr_trainer_base import CTRTrainerBase


TrainerType = TypeVar("TrainerType", bound=CTRTrainerBase)
ModelBuilder = Callable[[FeatureEncoder, Mapping[str, Any]], nn.Module]


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device: {requested}")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_run_directory(
    *, output_root: Path, experiment_name: str, run_name: str | None
) -> Path:
    suffix = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = output_root / f"{experiment_name}_{suffix}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def resolve_resume_checkpoint(
    project_root: Path, value: str | Path | None
) -> Path | None:
    if value is None:
        return None
    path = project_path(project_root, value)
    if path.is_dir():
        path = path / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint is missing: {path}")
    if path.name != "latest.pt":
        raise ValueError("resume from latest.pt, not the best-model checkpoint")
    return path


def create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"ads_training.{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def create_training_dataloader(
    *,
    dataset: Any,
    training_config: Mapping[str, Any],
    pin_memory: bool,
) -> tuple[Any, dict[str, Any]]:
    """Create the training loader and record its sampling distribution."""

    sampling_config = training_config.get("negative_sampling")
    if sampling_config is None:
        sampling_config = {"enabled": False}
    if not isinstance(sampling_config, Mapping):
        raise ValueError("training.negative_sampling must be a mapping")

    if bool(sampling_config.get("enabled", False)):
        sampling_summary = configure_negative_downsampling(
            dataset,
            negative_to_positive_ratio=float(
                sampling_config["negative_to_positive_ratio"]
            ),
            seed=int(sampling_config.get("seed", training_config["seed"])),
        )
    else:
        counts = dataset.metadata["counts"]
        original_samples = int(counts["samples"])
        positive_samples = int(counts["clicks"])
        sampling_summary = {
            "enabled": False,
            "strategy": "all_samples",
            "original_samples": original_samples,
            "original_positive_samples": positive_samples,
            "original_negative_samples": original_samples - positive_samples,
            "sampled_samples_per_epoch": original_samples,
            "negative_keep_probability": 1.0,
        }

    loader = create_dataloader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        pin_memory=pin_memory,
        columnar_batching=bool(training_config.get("columnar_batching", True)),
    )
    return loader, sampling_summary


def run_ctr_experiment(
    *,
    project_root: Path,
    config_path: str | Path,
    expected_model_name: str,
    model_builder: ModelBuilder,
    trainer_class: type[TrainerType],
    device_override: str | None,
    run_name: str | None,
    data_config_override: str | Path | None = None,
    experiment_name_override: str | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
    resume_from: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run train/validation/test while keeping model entrypoints explicit."""

    resolved_config_path = project_path(project_root, config_path)
    experiment_file = load_yaml(resolved_config_path)
    model_config = experiment_file.get("model")
    data_reference = experiment_file.get("data")
    training_config = experiment_file.get("training")
    experiment_config = experiment_file.get("experiment")
    for name, value in (
        ("model", model_config),
        ("data", data_reference),
        ("training", training_config),
        ("experiment", experiment_config),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"configuration section {name!r} must be a mapping")

    model_name = str(model_config.get("name", "")).lower()
    if model_name != expected_model_name:
        raise ValueError(
            f"expected model.name={expected_model_name!r}, got {model_name!r}"
        )

    training_config = dict(training_config)
    if batch_size_override is not None:
        if batch_size_override <= 0:
            raise ValueError("batch_size_override must be positive")
        training_config["batch_size"] = batch_size_override
    if num_workers_override is not None:
        if num_workers_override < 0:
            raise ValueError("num_workers_override cannot be negative")
        training_config["num_workers"] = num_workers_override

    selected_data_config = (
        data_config_override
        if data_config_override is not None
        else str(data_reference["config"])
    )
    data_config_path = project_path(project_root, selected_data_config)
    data_file = load_yaml(data_config_path)
    data_config = data_file.get("data")
    if not isinstance(data_config, Mapping):
        raise ValueError("data configuration must contain a data mapping")
    processed_dir = project_path(project_root, str(data_config["processed_dir"]))
    device = resolve_device(str(device_override or training_config["device"]))
    set_seed(int(training_config["seed"]))

    experiment_name = str(
        experiment_name_override or experiment_config["name"]
    )
    resume_checkpoint_path = resolve_resume_checkpoint(project_root, resume_from)
    if resume_checkpoint_path is not None:
        if run_name is not None:
            raise ValueError("--run-name cannot be combined with --resume-from")
        run_directory = resume_checkpoint_path.parent
    else:
        run_directory = build_run_directory(
            output_root=project_path(
                project_root, str(experiment_config["output_root"])
            ),
            experiment_name=experiment_name,
            run_name=run_name,
        )
    logger = create_logger(run_directory / "train.log")
    resolved_experiment_file = dict(experiment_file)
    resolved_experiment_file["data"] = {
        **dict(data_reference),
        "config": str(selected_data_config),
    }
    resolved_experiment_file["experiment"] = {
        **dict(experiment_config),
        "name": experiment_name,
    }
    resolved_experiment_file["training"] = training_config
    config_snapshot = run_directory / "config.yaml"
    if resume_checkpoint_path is None:
        config_snapshot.write_text(
            yaml.safe_dump(
                {"model_config": resolved_experiment_file, "data_config": data_file},
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    elif not config_snapshot.is_file():
        raise FileNotFoundError(
            f"resumed run is missing its configuration snapshot: {config_snapshot}"
        )
    logger.info("run_directory=%s", run_directory)
    logger.info(
        "model=%s device=%s torch=%s cuda=%s amp_requested=%s",
        model_name,
        device,
        torch.__version__,
        torch.version.cuda,
        training_config["amp"],
    )

    train_dataset: Any | None = None
    validation_dataset: Any | None = None
    test_dataset: Any | None = None
    try:
        dataset_seed = int(training_config["seed"])
        train_dataset = AdsDataset(processed_dir, split="train", seed=dataset_seed)
        validation_dataset = AdsDataset(
            processed_dir, split="validation", seed=dataset_seed
        )
        test_dataset = AdsDataset(processed_dir, split="test", seed=dataset_seed)
        batch_size = int(training_config["batch_size"])
        num_workers = int(training_config["num_workers"])
        pin_memory = bool(training_config["pin_memory"] and device.type == "cuda")
        train_loader, sampling_summary = create_training_dataloader(
            dataset=train_dataset,
            training_config=training_config,
            pin_memory=pin_memory,
        )
        logger.info("negative_sampling=%s", sampling_summary)
        validation_loader = create_dataloader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            columnar_batching=bool(
                training_config.get("columnar_batching", True)
            ),
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            columnar_batching=bool(
                training_config.get("columnar_batching", True)
            ),
        )

        encoder = FeatureEncoder.load(processed_dir / "vocab.json")
        model = model_builder(encoder, model_config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        gradient_clip_value = training_config.get("gradient_clip_norm")
        gradient_clip_norm = (
            None if gradient_clip_value is None else float(gradient_clip_value)
        )
        trainer = trainer_class(
            model=model,
            optimizer=optimizer,
            device=device,
            amp=bool(training_config["amp"]),
            gradient_clip_norm=gradient_clip_norm,
            logger=logger,
            negative_keep_probability=float(
                sampling_summary["negative_keep_probability"]
            ),
        )
        checkpoint_path = run_directory / "best.pt"
        latest_checkpoint_path = run_directory / "latest.pt"
        training_result = trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=int(training_config["epochs"]),
            early_stopping_patience=int(
                training_config["early_stopping_patience"]
            ),
            early_stopping_min_delta=float(
                training_config["early_stopping_min_delta"]
            ),
            checkpoint_path=checkpoint_path,
            latest_checkpoint_path=latest_checkpoint_path,
            resume_checkpoint_path=resume_checkpoint_path,
        )
        test_metrics = trainer.evaluate(test_loader)
        logger.info(
            "test_auc=%.6f test_gauc=%s test_log_loss=%.6f test_pr_auc=%.6f",
            test_metrics["auc"],
            (
                "n/a"
                if test_metrics["gauc"] is None
                else f"{test_metrics['gauc']:.6f}"
            ),
            test_metrics["log_loss"],
            test_metrics["pr_auc"],
        )
        if "prior_corrected" in test_metrics:
            corrected = test_metrics["prior_corrected"]
            logger.info(
                "corrected_test_auc=%.6f corrected_test_log_loss=%.6f "
                "corrected_test_pr_auc=%.6f "
                "corrected_test_prediction_mean=%.6f",
                corrected["auc"],
                corrected["log_loss"],
                corrected["pr_auc"],
                corrected["prediction_mean"],
            )

        result = {
            "model": model_name,
            "target": str(model_config.get("target", "click")),
            "model_hyperparameters": dict(model_config),
            "data_config": str(data_config_path),
            "processed_dir": str(processed_dir),
            "device": str(device),
            "amp_enabled": trainer.amp_enabled,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "dataset_sizes": {
                "train": len(train_dataset),
                "validation": len(validation_dataset),
                "test": len(test_dataset),
            },
            "negative_sampling": sampling_summary,
            "training": training_result,
            "test": test_metrics,
            "checkpoint": str(checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint_path),
            "resumed_from": (
                None
                if resume_checkpoint_path is None
                else str(resume_checkpoint_path)
            ),
        }
        write_json(run_directory / "metrics.json", result)
        return run_directory, result
    except Exception:
        logger.exception("experiment failed")
        raise
    finally:
        for dataset in (train_dataset, validation_dataset, test_dataset):
            if dataset is not None:
                dataset.close()
