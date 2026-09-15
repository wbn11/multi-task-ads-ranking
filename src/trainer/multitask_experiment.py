"""Shared orchestration for CTR/CVR/CTCVR multi-task experiments."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml
from torch import nn

from src.data.dataloader import create_dataloader
from src.data.dataset import AdsDataset
from src.data.feature_encoder import FeatureEncoder
from src.trainer.cli import merge_config_overrides
from src.trainer.ctr_experiment import (
    build_run_directory,
    create_logger,
    create_training_dataloader,
    load_yaml,
    project_path,
    resolve_resume_checkpoint,
    resolve_device,
    set_seed,
    write_json,
)
from src.trainer.multitask_trainer_base import MultiTaskTrainerBase


TrainerType = TypeVar("TrainerType", bound=MultiTaskTrainerBase)
ModelBuilder = Callable[[FeatureEncoder, Mapping[str, Any]], nn.Module]


def run_multitask_experiment(
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
    config_overrides: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run multi-task train/validation/test and persist one traceable run."""

    resolved_config_path = project_path(project_root, config_path)
    experiment_file = merge_config_overrides(
        load_yaml(resolved_config_path), config_overrides
    )
    sections = {
        name: experiment_file.get(name)
        for name in ("model", "loss", "data", "training", "experiment")
    }
    for name, value in sections.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"configuration section {name!r} must be a mapping")
    model_config = sections["model"]
    loss_config = sections["loss"]
    data_reference = sections["data"]
    training_config = sections["training"]
    experiment_config = sections["experiment"]

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
            loss_config=loss_config,
        )
        selection_task = str(training_config["selection_task"])
        selection_metric = str(training_config["selection_metric"])
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
            selection_task=selection_task,
            selection_metric=selection_metric,
            checkpoint_path=checkpoint_path,
            latest_checkpoint_path=latest_checkpoint_path,
            resume_checkpoint_path=resume_checkpoint_path,
        )
        test_metrics = trainer.evaluate(test_loader)
        logger.info(
            "test_ctr_auc=%.6f test_cvr_auc=%.6f test_ctcvr_auc=%.6f "
            "test_ctr_gauc=%s test_cvr_gauc=%s test_ctcvr_gauc=%s "
            "test_ctr_log_loss=%.6f test_cvr_log_loss=%.6f "
            "test_ctcvr_log_loss=%.6f",
            test_metrics["ctr"]["auc"],
            test_metrics["cvr"]["auc"],
            test_metrics["ctcvr"]["auc"],
            *(
                "n/a" if metrics["gauc"] is None else f"{metrics['gauc']:.6f}"
                for metrics in (
                    test_metrics["ctr"],
                    test_metrics["cvr"],
                    test_metrics["ctcvr"],
                )
            ),
            test_metrics["ctr"]["log_loss"],
            test_metrics["cvr"]["log_loss"],
            test_metrics["ctcvr"]["log_loss"],
        )

        result = {
            "model": model_name,
            "targets": list(model_config.get("targets", ("click", "conversion"))),
            "model_hyperparameters": dict(model_config),
            "loss_hyperparameters": dict(loss_config),
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
