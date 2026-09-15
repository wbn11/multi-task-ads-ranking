"""Shared single-task CTR training and evaluation implementation."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.calibration.prior import (
    correct_probabilities_for_negative_sampling,
    negative_sampling_logit_offset,
)
from src.data.dataset import stable_user_group_id
from src.metrics.binary import compute_binary_metrics
from src.metrics.calibration import compute_calibration_metrics
from src.metrics.gauc import compute_gauc
from src.trainer.checkpoint import (
    capture_random_state,
    restore_random_state,
    save_checkpoint_atomic,
    set_loader_iteration,
)


def move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move model inputs and labels while leaving string identifiers on CPU."""

    non_blocking = device.type == "cuda"
    features = {
        field_id: {
            name: tensor.to(device, non_blocking=non_blocking)
            for name, tensor in field.items()
        }
        for field_id, field in batch["features"].items()
    }
    return {
        "features": features,
        "click": batch["click"].to(device, non_blocking=non_blocking),
        "conversion": batch["conversion"].to(
            device, non_blocking=non_blocking
        ),
        "ctcvr": batch["ctcvr"].to(device, non_blocking=non_blocking),
    }


class CTRTrainerBase:
    """Train a CTR model exposing ``ctr_logit`` and ``ctr`` outputs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        amp: bool,
        gradient_clip_norm: float | None,
        logger: logging.Logger,
        negative_keep_probability: float = 1.0,
    ) -> None:
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or null")
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.amp_enabled = bool(amp and device.type == "cuda")
        self.gradient_clip_norm = gradient_clip_norm
        self.logger = logger
        self.negative_keep_probability = float(negative_keep_probability)
        negative_sampling_logit_offset(self.negative_keep_probability)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def train_epoch(self, loader: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        sample_count = 0
        started = time.perf_counter()

        for cpu_batch in loader:
            batch = move_batch_to_device(cpu_batch, self.device)
            labels = batch["click"]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                output = self.model(batch)
                loss = self.criterion(output["ctr_logit"], labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training produced a non-finite loss")

            self.scaler.scale(loss).backward()
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = int(labels.numel())
            loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("training DataLoader produced no samples")
        return {
            "loss": loss_sum / sample_count,
            "samples": float(sample_count),
            "seconds": time.perf_counter() - started,
        }

    @torch.no_grad()
    def collect_predictions(
        self,
        loader: Iterable[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return compact CPU labels, probabilities and user grouping IDs."""

        self.model.eval()
        dataset = getattr(loader, "dataset", None)
        expected_samples = len(dataset) if dataset is not None else None
        labels = (
            np.empty(expected_samples, dtype=np.uint8)
            if expected_samples is not None
            else None
        )
        probabilities = (
            np.empty(expected_samples, dtype=np.float32)
            if expected_samples is not None
            else None
        )
        user_group_ids = (
            np.empty(expected_samples, dtype=np.int64)
            if expected_samples is not None
            else None
        )
        labels_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []
        user_parts: list[np.ndarray] = []
        cursor = 0

        for cpu_batch in loader:
            cpu_labels = cpu_batch["click"].numpy().astype(np.uint8, copy=False)
            if "user_group_id" in cpu_batch:
                cpu_users = cpu_batch["user_group_id"].numpy().astype(
                    np.int64, copy=False
                )
            else:
                cpu_users = np.asarray(
                    [stable_user_group_id(value) for value in cpu_batch["user_id"]],
                    dtype=np.int64,
                )
            batch = move_batch_to_device(cpu_batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                output = self.model(batch)
            cpu_probabilities = output["ctr"].float().cpu().numpy()
            batch_size = int(cpu_labels.size)
            end = cursor + batch_size
            if labels is not None:
                if end > len(labels):
                    raise RuntimeError(
                        "evaluation produced more samples than dataset metadata"
                    )
                labels[cursor:end] = cpu_labels
                probabilities[cursor:end] = cpu_probabilities
                user_group_ids[cursor:end] = cpu_users
            else:
                labels_parts.append(cpu_labels.copy())
                probability_parts.append(cpu_probabilities.copy())
                user_parts.append(cpu_users.copy())
            cursor = end

        if cursor == 0:
            raise RuntimeError("evaluation DataLoader produced no samples")
        if labels is not None:
            return labels[:cursor], probabilities[:cursor], user_group_ids[:cursor]
        return (
            np.concatenate(labels_parts),
            np.concatenate(probability_parts),
            np.concatenate(user_parts),
        )

    def evaluate_predictions(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        user_ids: np.ndarray,
        *,
        calibration_num_bins: int | None = None,
        calibration_binning_strategy: str = "equal_frequency",
    ) -> dict[str, Any]:
        """Evaluate already collected predictions, including prior correction."""

        metrics = compute_binary_metrics(labels, probabilities)
        metrics.update(
            {
                "samples": float(labels.size),
                "positives": float(labels.sum()),
                "positive_rate": float(labels.mean()),
            }
        )
        gauc = compute_gauc(labels, probabilities, user_ids)
        metrics["gauc"] = gauc["value"]
        metrics["gauc_details"] = {
            key: value for key, value in gauc.items() if key != "value"
        }
        if calibration_num_bins is not None:
            metrics["calibration"] = compute_calibration_metrics(
                labels,
                probabilities,
                num_bins=calibration_num_bins,
                strategy=calibration_binning_strategy,
            )
        if self.negative_keep_probability < 1.0:
            corrected_probabilities = (
                correct_probabilities_for_negative_sampling(
                    probabilities,
                    negative_keep_probability=self.negative_keep_probability,
                )
            )
            corrected_metrics = compute_binary_metrics(
                labels,
                corrected_probabilities,
            )
            corrected_metrics.update(
                {
                    "samples": float(labels.size),
                    "positives": float(labels.sum()),
                    "positive_rate": float(labels.mean()),
                }
            )
            corrected_gauc = compute_gauc(
                labels,
                corrected_probabilities,
                user_ids,
            )
            corrected_metrics["gauc"] = corrected_gauc["value"]
            corrected_metrics["gauc_details"] = {
                key: value
                for key, value in corrected_gauc.items()
                if key != "value"
            }
            if calibration_num_bins is not None:
                corrected_metrics["calibration"] = (
                    compute_calibration_metrics(
                        labels,
                        corrected_probabilities,
                        num_bins=calibration_num_bins,
                        strategy=calibration_binning_strategy,
                    )
                )
            metrics["prior_corrected"] = corrected_metrics
            metrics["prior_correction"] = {
                "method": "uniform_negative_sampling_prior_correction",
                "negative_keep_probability": self.negative_keep_probability,
                "logit_offset": negative_sampling_logit_offset(
                    self.negative_keep_probability
                ),
            }
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Mapping[str, Any]],
        *,
        calibration_num_bins: int | None = None,
        calibration_binning_strategy: str = "equal_frequency",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        labels, probabilities, user_ids = self.collect_predictions(loader)
        metrics = self.evaluate_predictions(
            labels,
            probabilities,
            user_ids,
            calibration_num_bins=calibration_num_bins,
            calibration_binning_strategy=calibration_binning_strategy,
        )
        metrics["seconds"] = time.perf_counter() - started
        return metrics

    def fit(
        self,
        *,
        train_loader: Iterable[Mapping[str, Any]],
        validation_loader: Iterable[Mapping[str, Any]],
        epochs: int,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        checkpoint_path: Path,
        latest_checkpoint_path: Path,
        resume_checkpoint_path: Path | None = None,
    ) -> dict[str, Any]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")

        history: list[dict[str, Any]] = []
        best_auc = -float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        start_epoch = 1
        resumed_from_epoch = 0
        accumulated_seconds = 0.0
        if resume_checkpoint_path is not None:
            checkpoint = torch.load(
                resume_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            required = {
                "epoch",
                "model_state_dict",
                "optimizer_state_dict",
                "scaler_state_dict",
                "history",
                "best_auc",
                "best_epoch",
                "epochs_without_improvement",
            }
            missing = required - set(checkpoint)
            if missing:
                raise ValueError(
                    "resume checkpoint is missing training state: "
                    f"{sorted(missing)}"
                )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            history = list(checkpoint["history"])
            best_auc = float(checkpoint["best_auc"])
            best_epoch = int(checkpoint["best_epoch"])
            epochs_without_improvement = int(
                checkpoint["epochs_without_improvement"]
            )
            resumed_from_epoch = int(checkpoint["epoch"])
            start_epoch = resumed_from_epoch + 1
            accumulated_seconds = float(checkpoint.get("accumulated_seconds", 0.0))
            restore_random_state(checkpoint.get("random_state"))
            set_loader_iteration(train_loader, resumed_from_epoch)
            self.logger.info(
                "resumed training from epoch=%d checkpoint=%s",
                resumed_from_epoch,
                resume_checkpoint_path,
            )
        started = time.perf_counter()

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            validation_metrics = self.evaluate(validation_loader)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(record)
            self.logger.info(
                "epoch=%d train_loss=%.6f val_auc=%.6f val_gauc=%s "
                "val_log_loss=%.6f val_pr_auc=%.6f lr=%.6g "
                "train_seconds=%.2f val_seconds=%.2f",
                epoch,
                train_metrics["loss"],
                validation_metrics["auc"],
                (
                    "n/a"
                    if validation_metrics["gauc"] is None
                    else f"{validation_metrics['gauc']:.6f}"
                ),
                validation_metrics["log_loss"],
                validation_metrics["pr_auc"],
                learning_rate,
                train_metrics["seconds"],
                validation_metrics["seconds"],
            )
            if "prior_corrected" in validation_metrics:
                corrected = validation_metrics["prior_corrected"]
                self.logger.info(
                    "epoch=%d corrected_val_auc=%.6f "
                    "corrected_val_log_loss=%.6f corrected_val_pr_auc=%.6f "
                    "corrected_val_prediction_mean=%.6f",
                    epoch,
                    corrected["auc"],
                    corrected["log_loss"],
                    corrected["pr_auc"],
                    corrected["prediction_mean"],
                )

            improved = validation_metrics["auc"] > (
                best_auc + early_stopping_min_delta
            )
            should_stop = False
            if improved:
                best_auc = validation_metrics["auc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                )
                self.logger.info(
                    "saved best checkpoint: epoch=%d val_auc=%.6f path=%s",
                    epoch,
                    best_auc,
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    should_stop = True

            elapsed_seconds = accumulated_seconds + time.perf_counter() - started
            save_checkpoint_atomic(
                latest_checkpoint_path,
                {
                    "checkpoint_kind": "latest",
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "history": history,
                    "best_auc": best_auc,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "validation_metrics": dict(validation_metrics),
                    "accumulated_seconds": elapsed_seconds,
                    "random_state": capture_random_state(),
                },
            )
            self.logger.info(
                "saved resumable checkpoint: epoch=%d path=%s",
                epoch,
                latest_checkpoint_path,
            )
            if should_stop:
                self.logger.info(
                    "early stopping after epoch=%d; best_epoch=%d",
                    epoch,
                    best_epoch,
                )
                break

        if best_epoch == 0:
            raise RuntimeError("training finished without a valid checkpoint")
        self.load_checkpoint(checkpoint_path)
        return {
            "best_epoch": best_epoch,
            "best_validation_auc": best_auc,
            "epochs_completed": len(history),
            "resumed_from_epoch": resumed_from_epoch,
            "total_seconds": accumulated_seconds + time.perf_counter() - started,
            "history": history,
        }

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        validation_metrics: Mapping[str, Any],
    ) -> None:
        save_checkpoint_atomic(
            path,
            {
                "checkpoint_kind": "best",
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "validation_metrics": dict(validation_metrics),
                "random_state": capture_random_state(),
            },
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
