"""Evaluate a saved single-task CTR run without retraining it."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.platt import PlattCalibrator
from src.calibration.prior import correct_probabilities_for_negative_sampling
from src.data.dataloader import create_dataloader
from src.data.dataset import AdsDataset
from src.data.feature_encoder import FeatureEncoder
from src.metrics.binary import compute_binary_metrics
from src.metrics.calibration import compute_calibration_metrics
from src.metrics.gauc import compute_gauc
from src.trainer.ctr_experiment import (
    ModelBuilder,
    TrainerType,
    load_yaml,
    project_path,
    resolve_device,
    set_seed,
    write_json,
)


def evaluate_ctr_run(
    *,
    project_root: Path,
    run_directory: str | Path,
    expected_model_name: str,
    model_builder: ModelBuilder,
    trainer_class: type[TrainerType],
    device_override: str | None,
    processed_dir_override: str | Path | None,
    calibration_num_bins: int,
    calibration_binning_strategy: str,
    platt_l2_regularization: float,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate raw, prior-corrected, Platt and isotonic CTR probabilities."""

    if calibration_num_bins <= 1:
        raise ValueError("calibration_num_bins must be greater than one")
    if calibration_binning_strategy not in ("equal_frequency", "equal_width"):
        raise ValueError(
            "calibration_binning_strategy must be equal_frequency or equal_width"
        )

    def evaluate_probability_array(
        labels: np.ndarray,
        probabilities: np.ndarray,
        user_ids: np.ndarray,
    ) -> dict[str, Any]:
        metrics = compute_binary_metrics(labels, probabilities)
        gauc = compute_gauc(labels, probabilities, user_ids)
        metrics.update(
            {
                "samples": float(labels.size),
                "positives": float(labels.sum()),
                "positive_rate": float(labels.mean()),
                "calibration": compute_calibration_metrics(
                    labels,
                    probabilities,
                    num_bins=calibration_num_bins,
                    strategy=calibration_binning_strategy,
                ),
                "gauc": gauc["value"],
                "gauc_details": {
                    key: value for key, value in gauc.items() if key != "value"
                },
            }
        )
        return metrics

    resolved_run_directory = project_path(project_root, run_directory)
    saved_config = load_yaml(resolved_run_directory / "config.yaml")
    experiment_file = saved_config.get("model_config")
    data_file = saved_config.get("data_config")
    if not isinstance(experiment_file, Mapping):
        raise ValueError("saved config must contain model_config")
    if not isinstance(data_file, Mapping):
        raise ValueError("saved config must contain data_config")

    model_config = experiment_file.get("model")
    training_config = experiment_file.get("training")
    data_config = data_file.get("data")
    for name, value in (
        ("model", model_config),
        ("training", training_config),
        ("data", data_config),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"saved {name} configuration must be a mapping")
    model_name = str(model_config.get("name", "")).lower()
    if model_name != expected_model_name:
        raise ValueError(
            f"expected model.name={expected_model_name!r}, got {model_name!r}"
        )

    metrics_path = resolved_run_directory / "metrics.json"
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    sampling_summary = saved_metrics.get("negative_sampling")
    if not isinstance(sampling_summary, Mapping):
        raise ValueError("saved metrics must contain negative_sampling")
    negative_keep_probability = float(
        sampling_summary["negative_keep_probability"]
    )

    processed_dir = project_path(
        project_root,
        (
            processed_dir_override
            if processed_dir_override is not None
            else str(data_config["processed_dir"])
        ),
    )
    device = resolve_device(str(device_override or training_config["device"]))
    set_seed(int(training_config["seed"]))
    checkpoint_path = resolved_run_directory / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")

    validation_dataset: Any | None = None
    test_dataset: Any | None = None
    try:
        dataset_seed = int(training_config["seed"])
        validation_dataset = AdsDataset(
            processed_dir, split="validation", seed=dataset_seed
        )
        test_dataset = AdsDataset(
            processed_dir, split="test", seed=dataset_seed
        )
        batch_size = int(training_config["batch_size"])
        num_workers = int(training_config["num_workers"])
        pin_memory = bool(training_config["pin_memory"] and device.type == "cuda")
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
        logger = logging.getLogger(
            f"ads_evaluation.{resolved_run_directory.name}"
        )
        logger.addHandler(logging.NullHandler())
        trainer = trainer_class(
            model=model,
            optimizer=optimizer,
            device=device,
            amp=bool(training_config["amp"]),
            gradient_clip_norm=gradient_clip_norm,
            logger=logger,
            negative_keep_probability=negative_keep_probability,
        )
        checkpoint = trainer.load_checkpoint(checkpoint_path)
        validation_started = time.perf_counter()
        validation_labels, validation_probabilities, validation_user_ids = (
            trainer.collect_predictions(validation_loader)
        )
        validation_seconds = time.perf_counter() - validation_started
        test_started = time.perf_counter()
        test_labels, test_probabilities, test_user_ids = (
            trainer.collect_predictions(test_loader)
        )
        test_seconds = time.perf_counter() - test_started
        validation_metrics = trainer.evaluate_predictions(
            validation_labels,
            validation_probabilities,
            validation_user_ids,
            calibration_num_bins=calibration_num_bins,
            calibration_binning_strategy=calibration_binning_strategy,
        )
        validation_metrics["seconds"] = validation_seconds
        test_metrics = trainer.evaluate_predictions(
            test_labels,
            test_probabilities,
            test_user_ids,
            calibration_num_bins=calibration_num_bins,
            calibration_binning_strategy=calibration_binning_strategy,
        )
        test_metrics["seconds"] = test_seconds

        validation_calibration_input = (
            correct_probabilities_for_negative_sampling(
                validation_probabilities,
                negative_keep_probability=negative_keep_probability,
            )
        )
        test_calibration_input = correct_probabilities_for_negative_sampling(
            test_probabilities,
            negative_keep_probability=negative_keep_probability,
        )
        calibration_input_stage = (
            "prior_corrected"
            if negative_keep_probability < 1.0
            else "raw_probability"
        )

        platt = PlattCalibrator(
            l2_regularization=platt_l2_regularization,
        )
        platt_started = time.perf_counter()
        platt.fit(validation_calibration_input, validation_labels)
        platt_fit_seconds = time.perf_counter() - platt_started
        platt_validation = platt.transform(validation_calibration_input)
        platt_test = platt.transform(test_calibration_input)

        isotonic = IsotonicCalibrator()
        isotonic_started = time.perf_counter()
        isotonic.fit(validation_calibration_input, validation_labels)
        isotonic_fit_seconds = time.perf_counter() - isotonic_started
        isotonic_validation = isotonic.transform(validation_calibration_input)
        isotonic_test = isotonic.transform(test_calibration_input)

        calibrator_path = resolved_run_directory / "calibrators.json"
        calibrator_payload = {
            "schema_version": 1,
            "source_run_directory": str(resolved_run_directory),
            "checkpoint": str(checkpoint_path),
            "fit_split": "validation",
            "test_used_for_fit": False,
            "input_stage": calibration_input_stage,
            "negative_keep_probability": negative_keep_probability,
            "calibrators": {
                "platt": platt.state_dict(),
                "isotonic": isotonic.state_dict(),
            },
        }
        write_json(calibrator_path, calibrator_payload)

        result = {
            "model": model_name,
            "source_run_directory": str(resolved_run_directory),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "device": str(device),
            "processed_dir": str(processed_dir),
            "negative_sampling": dict(sampling_summary),
            "calibration_evaluation": {
                "num_bins": calibration_num_bins,
                "binning_strategy": calibration_binning_strategy,
            },
            "validation": validation_metrics,
            "test": test_metrics,
            "learned_calibration": {
                "fit_split": "validation",
                "test_used_for_fit": False,
                "input_stage": calibration_input_stage,
                "calibrator_path": str(calibrator_path),
                "platt": {
                    "fit_seconds": platt_fit_seconds,
                    "parameters": {
                        "slope": platt.slope_,
                        "intercept": platt.intercept_,
                        "converged": platt.converged_,
                        "iterations": platt.iterations_,
                        "l2_regularization": platt.l2_regularization,
                    },
                    "validation": evaluate_probability_array(
                        validation_labels,
                        platt_validation,
                        validation_user_ids,
                    ),
                    "test": evaluate_probability_array(
                        test_labels,
                        platt_test,
                        test_user_ids,
                    ),
                },
                "isotonic": {
                    "fit_seconds": isotonic_fit_seconds,
                    "parameters": {
                        "num_blocks": isotonic.num_blocks_,
                        "num_thresholds": int(isotonic.x_thresholds_.size),
                        "interpolation": "linear_between_pooled_blocks",
                    },
                    "validation": evaluate_probability_array(
                        validation_labels,
                        isotonic_validation,
                        validation_user_ids,
                    ),
                    "test": evaluate_probability_array(
                        test_labels,
                        isotonic_test,
                        test_user_ids,
                    ),
                },
            },
        }
        output_path = resolved_run_directory / "evaluation_calibrated.json"
        write_json(output_path, result)
        return output_path, result
    finally:
        for dataset in (validation_dataset, test_dataset):
            if dataset is not None:
                dataset.close()
