"""Post-hoc probability evaluation for saved multi-task model runs."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.platt import PlattCalibrator
from src.calibration.prior import correct_probabilities_for_negative_sampling
from src.data.dataloader import create_dataloader
from src.data.dataset import AdsDataset, stable_user_group_id
from src.data.feature_encoder import FeatureEncoder
from src.metrics.calibration import compute_calibration_metrics
from src.metrics.multitask import compute_multitask_metrics
from src.models.factory import build_multitask_model
from src.trainer.ctr_experiment import (
    load_yaml,
    project_path,
    resolve_device,
    set_seed,
    write_json,
)
from src.trainer.ctr_trainer_base import move_batch_to_device


ArrayBundle = dict[str, np.ndarray]
ProbabilityBundle = dict[str, np.ndarray]


def correct_multitask_sampling_prior(
    probabilities: Mapping[str, np.ndarray],
    *,
    negative_keep_probability: float,
) -> ProbabilityBundle:
    """Correct CTR prior and rebuild CTCVR without inventing a CVR formula.

    Non-click downsampling is class-conditional for CTR, so its analytical
    prior correction is exact under uniform sampling. No analytical correction
    is applied to CVR: clicked rows are fully retained, while any indirect ESMM
    bias is handled by validation-fitted calibration. CTCVR is rebuilt from the
    two heads to preserve ``pCTCVR = pCTR * pCVR``; it is intentionally not
    given an invalid binary-class correction because its negative class mixes
    sampled non-clicks and fully retained clicked non-conversions.
    """

    ctr = correct_probabilities_for_negative_sampling(
        probabilities["ctr"],
        negative_keep_probability=negative_keep_probability,
    )
    cvr = np.asarray(probabilities["cvr"], dtype=np.float64)
    if ctr.shape != cvr.shape:
        raise ValueError("CTR and CVR probabilities must have identical shapes")
    return {
        "ctr": np.asarray(ctr, dtype=np.float64),
        "cvr": cvr,
        "ctcvr": np.asarray(ctr, dtype=np.float64) * cvr,
    }


def evaluate_multitask_probability_bundle(
    arrays: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    *,
    calibration_num_bins: int,
    calibration_binning_strategy: str,
) -> dict[str, Any]:
    """Evaluate ranking, likelihood and calibration on each task's space."""

    metrics = compute_multitask_metrics(
        click_labels=arrays["click"],
        conversion_labels=arrays["conversion"],
        ctcvr_labels=arrays["ctcvr"],
        ctr_probabilities=probabilities["ctr"],
        cvr_probabilities=probabilities["cvr"],
        ctcvr_probabilities=probabilities["ctcvr"],
        user_ids=arrays["user"],
    )
    clicked = arrays["click"] == 1
    task_views = {
        "ctr": (arrays["click"], probabilities["ctr"]),
        "cvr": (
            arrays["conversion"][clicked],
            probabilities["cvr"][clicked],
        ),
        "ctcvr": (arrays["ctcvr"], probabilities["ctcvr"]),
    }
    for task, (labels, task_probabilities) in task_views.items():
        metrics[task]["calibration"] = compute_calibration_metrics(
            labels,
            task_probabilities,
            num_bins=calibration_num_bins,
            strategy=calibration_binning_strategy,
        )
    return metrics


@torch.no_grad()
def collect_multitask_predictions(
    *,
    model: nn.Module,
    loader: Any,
    device: torch.device,
    amp: bool,
) -> ArrayBundle:
    """Collect compact labels, user groups and probabilities for one split."""

    model.eval()
    dataset = getattr(loader, "dataset", None)
    expected_samples = len(dataset) if dataset is not None else None
    if expected_samples is None:
        raise ValueError("evaluation loader must expose a sized dataset")
    arrays: ArrayBundle = {
        "click": np.empty(expected_samples, dtype=np.uint8),
        "conversion": np.empty(expected_samples, dtype=np.uint8),
        "ctcvr": np.empty(expected_samples, dtype=np.uint8),
        "user": np.empty(expected_samples, dtype=np.int64),
        "ctr": np.empty(expected_samples, dtype=np.float32),
        "cvr": np.empty(expected_samples, dtype=np.float32),
        "predicted_ctcvr": np.empty(expected_samples, dtype=np.float32),
    }
    cursor = 0
    amp_enabled = bool(amp and device.type == "cuda")
    for cpu_batch in loader:
        click = cpu_batch["click"].numpy().astype(np.uint8, copy=False)
        batch_size = int(click.size)
        end = cursor + batch_size
        if end > expected_samples:
            raise RuntimeError("evaluation produced more rows than dataset metadata")
        arrays["click"][cursor:end] = click
        arrays["conversion"][cursor:end] = (
            cpu_batch["conversion"].numpy().astype(np.uint8, copy=False)
        )
        arrays["ctcvr"][cursor:end] = (
            cpu_batch["ctcvr"].numpy().astype(np.uint8, copy=False)
        )
        if "user_group_id" in cpu_batch:
            user = cpu_batch["user_group_id"].numpy().astype(np.int64, copy=False)
        else:
            user = np.asarray(
                [stable_user_group_id(value) for value in cpu_batch["user_id"]],
                dtype=np.int64,
            )
        arrays["user"][cursor:end] = user

        batch = move_batch_to_device(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(batch)
        for output_name, array_name in (
            ("ctr", "ctr"),
            ("cvr", "cvr"),
            ("ctcvr", "predicted_ctcvr"),
        ):
            values = outputs[output_name].float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"model produced non-finite {output_name} values")
            arrays[array_name][cursor:end] = values
        cursor = end
    if cursor == 0:
        raise RuntimeError("evaluation DataLoader produced no samples")
    return {name: values[:cursor] for name, values in arrays.items()}


def _raw_probabilities(arrays: Mapping[str, np.ndarray]) -> ProbabilityBundle:
    return {
        "ctr": np.asarray(arrays["ctr"], dtype=np.float64),
        "cvr": np.asarray(arrays["cvr"], dtype=np.float64),
        "ctcvr": np.asarray(arrays["predicted_ctcvr"], dtype=np.float64),
    }


def _fit_headwise_calibrators(
    calibrator_class: type[PlattCalibrator] | type[IsotonicCalibrator],
    arrays: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    **kwargs: Any,
) -> dict[str, Any]:
    clicked = arrays["click"] == 1
    ctr_calibrator = calibrator_class(**kwargs).fit(
        probabilities["ctr"], arrays["click"]
    )
    cvr_calibrator = calibrator_class(**kwargs).fit(
        probabilities["cvr"][clicked], arrays["conversion"][clicked]
    )
    return {"ctr": ctr_calibrator, "cvr": cvr_calibrator}


def _transform_headwise(
    calibrators: Mapping[str, Any],
    probabilities: Mapping[str, np.ndarray],
) -> ProbabilityBundle:
    ctr = calibrators["ctr"].transform(probabilities["ctr"])
    cvr = calibrators["cvr"].transform(probabilities["cvr"])
    return {"ctr": ctr, "cvr": cvr, "ctcvr": ctr * cvr}


def _calibrator_states(calibrators: Mapping[str, Any]) -> dict[str, Any]:
    return {
        task: calibrator.state_dict() for task, calibrator in calibrators.items()
    }


def evaluate_multitask_run(
    *,
    project_root: Path,
    run_directory: str | Path,
    device_override: str | None,
    processed_dir_override: str | Path | None,
    batch_size_override: int | None,
    num_workers_override: int | None,
    calibration_num_bins: int,
    calibration_binning_strategy: str,
    platt_l2_regularization: float,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate raw, prior-corrected and headwise-calibrated probabilities."""

    if calibration_num_bins <= 1:
        raise ValueError("calibration_num_bins must be greater than one")
    if calibration_binning_strategy not in ("equal_frequency", "equal_width"):
        raise ValueError(
            "calibration_binning_strategy must be equal_frequency or equal_width"
        )
    if platt_l2_regularization < 0.0:
        raise ValueError("platt_l2_regularization cannot be negative")
    resolved_run = project_path(project_root, run_directory)
    saved_config = load_yaml(resolved_run / "config.yaml")
    experiment_file = saved_config.get("model_config")
    data_file = saved_config.get("data_config")
    if not isinstance(experiment_file, Mapping) or not isinstance(data_file, Mapping):
        raise ValueError("saved config requires model_config and data_config mappings")
    model_config = experiment_file.get("model")
    training_config = experiment_file.get("training")
    data_config = data_file.get("data")
    if not all(
        isinstance(value, Mapping)
        for value in (model_config, training_config, data_config)
    ):
        raise ValueError("saved model, training and data sections must be mappings")

    saved_metrics = json.loads(
        (resolved_run / "metrics.json").read_text(encoding="utf-8")
    )
    sampling_summary = saved_metrics.get("negative_sampling")
    if not isinstance(sampling_summary, Mapping):
        raise ValueError("saved metrics must contain negative_sampling")
    negative_keep_probability = float(
        sampling_summary["negative_keep_probability"]
    )
    processed_dir = project_path(
        project_root,
        processed_dir_override or str(data_config["processed_dir"]),
    )
    device = resolve_device(str(device_override or training_config["device"]))
    seed = int(training_config["seed"])
    set_seed(seed)
    batch_size = int(
        training_config["batch_size"]
        if batch_size_override is None
        else batch_size_override
    )
    num_workers = int(
        training_config["num_workers"]
        if num_workers_override is None
        else num_workers_override
    )
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    checkpoint_path = resolved_run / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")

    validation_dataset: AdsDataset | None = None
    test_dataset: AdsDataset | None = None
    try:
        validation_dataset = AdsDataset(processed_dir, split="validation", seed=seed)
        test_dataset = AdsDataset(processed_dir, split="test", seed=seed)
        pin_memory = bool(training_config["pin_memory"] and device.type == "cuda")
        loader_arguments = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "columnar_batching": bool(
                training_config.get("columnar_batching", True)
            ),
        }
        validation_loader = create_dataloader(validation_dataset, **loader_arguments)
        test_loader = create_dataloader(test_dataset, **loader_arguments)
        encoder = FeatureEncoder.load(processed_dir / "vocab.json")
        model = build_multitask_model(encoder, model_config).to(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        validation_started = time.perf_counter()
        validation_arrays = collect_multitask_predictions(
            model=model,
            loader=validation_loader,
            device=device,
            amp=bool(training_config["amp"]),
        )
        validation_seconds = time.perf_counter() - validation_started
        test_started = time.perf_counter()
        test_arrays = collect_multitask_predictions(
            model=model,
            loader=test_loader,
            device=device,
            amp=bool(training_config["amp"]),
        )
        test_seconds = time.perf_counter() - test_started

        raw_validation = _raw_probabilities(validation_arrays)
        raw_test = _raw_probabilities(test_arrays)
        base_validation = correct_multitask_sampling_prior(
            raw_validation,
            negative_keep_probability=negative_keep_probability,
        )
        base_test = correct_multitask_sampling_prior(
            raw_test,
            negative_keep_probability=negative_keep_probability,
        )

        metric_arguments = {
            "calibration_num_bins": calibration_num_bins,
            "calibration_binning_strategy": calibration_binning_strategy,
        }
        raw_metrics = {
            "validation": evaluate_multitask_probability_bundle(
                validation_arrays, raw_validation, **metric_arguments
            ),
            "test": evaluate_multitask_probability_bundle(
                test_arrays, raw_test, **metric_arguments
            ),
        }
        raw_metrics["validation"]["seconds"] = validation_seconds
        raw_metrics["test"]["seconds"] = test_seconds

        prior_metrics = None
        if negative_keep_probability < 1.0:
            prior_metrics = {
                "validation": evaluate_multitask_probability_bundle(
                    validation_arrays, base_validation, **metric_arguments
                ),
                "test": evaluate_multitask_probability_bundle(
                    test_arrays, base_test, **metric_arguments
                ),
            }

        fitted: dict[str, dict[str, Any]] = {}
        for method, calibrator_class, kwargs in (
            (
                "platt",
                PlattCalibrator,
                {"l2_regularization": platt_l2_regularization},
            ),
            ("isotonic", IsotonicCalibrator, {}),
        ):
            fit_started = time.perf_counter()
            calibrators = _fit_headwise_calibrators(
                calibrator_class,
                validation_arrays,
                base_validation,
                **kwargs,
            )
            fit_seconds = time.perf_counter() - fit_started
            validation_probabilities = _transform_headwise(
                calibrators, base_validation
            )
            test_probabilities = _transform_headwise(calibrators, base_test)
            fitted[method] = {
                "fit_seconds": fit_seconds,
                "calibrators": _calibrator_states(calibrators),
                "validation": evaluate_multitask_probability_bundle(
                    validation_arrays,
                    validation_probabilities,
                    **metric_arguments,
                ),
                "test": evaluate_multitask_probability_bundle(
                    test_arrays,
                    test_probabilities,
                    **metric_arguments,
                ),
            }

        calibrator_path = resolved_run / "multitask_calibrators.json"
        write_json(
            calibrator_path,
            {
                "schema_version": 1,
                "source_run_directory": str(resolved_run),
                "checkpoint": str(checkpoint_path),
                "fit_split": "validation",
                "test_used_for_fit": False,
                "probability_identity": "ctcvr=ctr*cvr",
                "input_stage": (
                    "ctr_prior_corrected"
                    if negative_keep_probability < 1.0
                    else "raw_heads"
                ),
                "negative_keep_probability": negative_keep_probability,
                "methods": {
                    method: details["calibrators"]
                    for method, details in fitted.items()
                },
            },
        )
        result = {
            "model": str(model_config["name"]),
            "source_run_directory": str(resolved_run),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "device": str(device),
            "processed_dir": str(processed_dir),
            "negative_sampling": dict(sampling_summary),
            "calibration_evaluation": {
                "num_bins": calibration_num_bins,
                "binning_strategy": calibration_binning_strategy,
                "fit_split": "validation",
                "test_used_for_fit": False,
                "headwise_calibration": True,
                "probability_identity": "ctcvr=ctr*cvr",
            },
            "raw": raw_metrics,
            "prior_corrected": prior_metrics,
            "learned_calibration": fitted,
            "calibrator_path": str(calibrator_path),
        }
        output_path = resolved_run / "evaluation_calibrated_multitask.json"
        write_json(output_path, result)
        return output_path, result
    finally:
        for dataset in (validation_dataset, test_dataset):
            if dataset is not None:
                dataset.close()
