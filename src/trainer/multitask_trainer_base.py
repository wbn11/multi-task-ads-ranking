"""Shared training loop for CTR/CVR/CTCVR multi-task models."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.metrics.multitask import compute_multitask_metrics
from src.data.dataset import stable_user_group_id
from src.trainer.ctr_trainer_base import move_batch_to_device
from src.trainer.checkpoint import (
    capture_random_state,
    restore_random_state,
    save_checkpoint_atomic,
    set_loader_iteration,
)


class MultiTaskTrainerBase:
    """Train models exposing CTR/CVR logits and CTR/CVR/CTCVR probabilities."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        loss_component_scopes: Mapping[str, str],
        loss_component_weights: Mapping[str, float],
        device: torch.device,
        amp: bool,
        gradient_clip_norm: float | None,
        logger: logging.Logger,
        loss_count_metrics: Sequence[str] = (),
        loss_component_count_metrics: Mapping[str, str] | None = None,
    ) -> None:
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or null")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        if not loss_component_scopes:
            raise ValueError("loss_component_scopes cannot be empty")
        if set(loss_component_scopes) != set(loss_component_weights):
            raise ValueError(
                "loss component scopes and weights must have identical keys"
            )
        invalid_scopes = set(loss_component_scopes.values()) - {
            "all",
            "clicked",
        }
        if invalid_scopes:
            raise ValueError(
                f"unsupported loss component scopes: {sorted(invalid_scopes)}"
            )
        if any(float(weight) <= 0.0 for weight in loss_component_weights.values()):
            raise ValueError("loss component weights must be positive")
        self.loss_component_scopes = dict(loss_component_scopes)
        self.loss_component_weights = {
            name: float(weight) for name, weight in loss_component_weights.items()
        }
        if len(set(loss_count_metrics)) != len(loss_count_metrics):
            raise ValueError("loss_count_metrics must be unique")
        self.loss_count_metrics = tuple(loss_count_metrics)
        self.loss_component_count_metrics = dict(
            loss_component_count_metrics or {}
        )
        unknown_counted_components = (
            set(self.loss_component_count_metrics)
            - set(self.loss_component_scopes)
        )
        if unknown_counted_components:
            raise ValueError(
                "loss component count metrics reference unknown components: "
                f"{sorted(unknown_counted_components)}"
            )
        if any(
            not isinstance(metric_name, str) or not metric_name
            for metric_name in self.loss_component_count_metrics.values()
        ):
            raise ValueError("loss component count metric names must be non-empty")
        self.device = device
        self.amp_enabled = bool(amp and device.type == "cuda")
        self.gradient_clip_norm = gradient_clip_norm
        self.logger = logger
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def _evaluation_forward(
        self,
        batch: Mapping[str, Any],
    ) -> Mapping[str, Tensor]:
        return self.model(batch)

    def _create_evaluation_diagnostics(self) -> Any:
        return None

    def _update_evaluation_diagnostics(
        self,
        state: Any,
        outputs: Mapping[str, Tensor],
    ) -> None:
        del state, outputs

    def _finalize_evaluation_diagnostics(self, state: Any) -> dict[str, Any]:
        del state
        return {}

    def train_epoch(self, loader: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        self.model.train()
        component_sums = {
            name: 0.0 for name in self.loss_component_scopes
        }
        component_counts = {
            name: 0 for name in self.loss_component_scopes
        }
        count_metric_sums = {
            name: torch.zeros((), dtype=torch.float64, device=self.device)
            for name in self.loss_count_metrics
        }
        sample_count = 0
        clicked_count = 0
        conversion_count = 0
        started = time.perf_counter()

        for cpu_batch in loader:
            batch = move_batch_to_device(cpu_batch, self.device)
            batch_size = int(batch["click"].numel())
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                outputs = self.model(batch)
                losses = self.criterion(outputs, batch)
                loss = losses["loss"]
            if not isinstance(loss, Tensor) or not bool(torch.isfinite(loss)):
                raise RuntimeError("training produced a non-finite loss")

            self.scaler.scale(loss).backward()
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_clicked_count = int(batch["click"].sum().item())
            scope_counts = {
                "all": batch_size,
                "clicked": batch_clicked_count,
            }
            for name, scope in self.loss_component_scopes.items():
                component = losses.get(name)
                if not isinstance(component, Tensor) or component.ndim != 0:
                    raise RuntimeError(
                        f"criterion must return scalar tensor {name!r}"
                    )
                if not bool(torch.isfinite(component)):
                    raise RuntimeError(
                        f"training produced non-finite component {name!r}"
                    )
                count_metric_name = self.loss_component_count_metrics.get(name)
                if count_metric_name is None:
                    normalizer = scope_counts[scope]
                else:
                    count_value = losses.get(count_metric_name)
                    if (
                        not isinstance(count_value, Tensor)
                        or count_value.ndim != 0
                        or not bool(torch.isfinite(count_value))
                    ):
                        raise RuntimeError(
                            "criterion must return finite scalar count tensor "
                            f"{count_metric_name!r}"
                        )
                    normalizer = int(count_value.detach().item())
                    if normalizer < 0:
                        raise RuntimeError(
                            f"criterion returned negative count {count_metric_name!r}"
                        )
                component_sums[name] += (
                    float(component.detach().item()) * normalizer
                )
                component_counts[name] += normalizer
            for name, total in count_metric_sums.items():
                value = losses.get(name)
                if not isinstance(value, Tensor) or value.ndim != 0:
                    raise RuntimeError(
                        f"criterion must return scalar count tensor {name!r}"
                    )
                if not bool(torch.isfinite(value)):
                    raise RuntimeError(
                        f"training produced non-finite count metric {name!r}"
                    )
                total.add_(value.detach().to(dtype=torch.float64))
            sample_count += batch_size
            clicked_count += batch_clicked_count
            conversion_count += int(batch["conversion"].sum().item())

        if sample_count == 0:
            raise RuntimeError("training DataLoader produced no samples")
        empty_components = [
            name for name, count in component_counts.items() if count == 0
        ]
        if empty_components:
            raise RuntimeError(
                "training epoch has no samples for loss components: "
                f"{empty_components}"
            )
        component_metrics = {
            name: component_sums[name] / component_counts[name]
            for name in self.loss_component_scopes
        }
        objective = sum(
            self.loss_component_weights[name] * component_metrics[name]
            for name in self.loss_component_scopes
        )
        return {
            "loss": objective,
            **component_metrics,
            **{
                name: float(value.item())
                for name, value in count_metric_sums.items()
            },
            "samples": float(sample_count),
            "clicked_samples": float(clicked_count),
            "conversion_positives": float(conversion_count),
            "seconds": time.perf_counter() - started,
        }

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        self.model.eval()
        dataset = getattr(loader, "dataset", None)
        expected_samples = len(dataset) if dataset is not None else None
        compact_arrays = (
            {
                "click": np.empty(expected_samples, dtype=np.uint8),
                "conversion": np.empty(expected_samples, dtype=np.uint8),
                "ctcvr": np.empty(expected_samples, dtype=np.uint8),
                "ctr": np.empty(expected_samples, dtype=np.float32),
                "cvr": np.empty(expected_samples, dtype=np.float32),
                "predicted_ctcvr": np.empty(expected_samples, dtype=np.float32),
                "user": np.empty(expected_samples, dtype=np.int64),
            }
            if expected_samples is not None
            else None
        )
        parts: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "click",
                "conversion",
                "ctcvr",
                "ctr",
                "cvr",
                "predicted_ctcvr",
                "user",
            )
        }
        cursor = 0
        diagnostic_state = self._create_evaluation_diagnostics()
        started = time.perf_counter()

        for cpu_batch in loader:
            cpu_values = {
                "click": cpu_batch["click"].numpy().astype(np.uint8, copy=False),
                "conversion": cpu_batch["conversion"].numpy().astype(
                    np.uint8, copy=False
                ),
                "ctcvr": cpu_batch["ctcvr"].numpy().astype(np.uint8, copy=False),
            }
            if "user_group_id" in cpu_batch:
                cpu_values["user"] = cpu_batch["user_group_id"].numpy().astype(
                    np.int64, copy=False
                )
            else:
                cpu_values["user"] = np.asarray(
                    [stable_user_group_id(value) for value in cpu_batch["user_id"]],
                    dtype=np.int64,
                )
            batch = move_batch_to_device(cpu_batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                outputs = self._evaluation_forward(batch)
            self._update_evaluation_diagnostics(diagnostic_state, outputs)
            cpu_values.update(
                {
                    "ctr": outputs["ctr"].float().cpu().numpy(),
                    "cvr": outputs["cvr"].float().cpu().numpy(),
                    "predicted_ctcvr": outputs["ctcvr"].float().cpu().numpy(),
                }
            )
            batch_size = int(cpu_values["click"].size)
            end = cursor + batch_size
            if compact_arrays is not None:
                if end > expected_samples:
                    raise RuntimeError(
                        "evaluation produced more samples than dataset metadata"
                    )
                for name, values in cpu_values.items():
                    compact_arrays[name][cursor:end] = values
            else:
                for name, values in cpu_values.items():
                    parts[name].append(values.copy())
            cursor = end

        if cursor == 0:
            raise RuntimeError("evaluation DataLoader produced no samples")
        arrays = (
            {name: values[:cursor] for name, values in compact_arrays.items()}
            if compact_arrays is not None
            else {name: np.concatenate(values) for name, values in parts.items()}
        )
        metrics = compute_multitask_metrics(
            click_labels=arrays["click"],
            conversion_labels=arrays["conversion"],
            ctcvr_labels=arrays["ctcvr"],
            ctr_probabilities=arrays["ctr"],
            cvr_probabilities=arrays["cvr"],
            ctcvr_probabilities=arrays["predicted_ctcvr"],
            user_ids=arrays["user"],
        )
        metrics.update(self._finalize_evaluation_diagnostics(diagnostic_state))
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
        selection_task: str,
        selection_metric: str,
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
        if selection_task not in ("ctr", "cvr", "ctcvr"):
            raise ValueError("selection_task must be ctr, cvr or ctcvr")
        if selection_metric not in ("auc", "pr_auc"):
            raise ValueError("selection_metric must be auc or pr_auc")

        history: list[dict[str, Any]] = []
        best_score = -float("inf")
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
                "best_score",
                "best_epoch",
                "epochs_without_improvement",
                "selection_task",
                "selection_metric",
            }
            missing = required - set(checkpoint)
            if missing:
                raise ValueError(
                    "resume checkpoint is missing training state: "
                    f"{sorted(missing)}"
                )
            if checkpoint["selection_task"] != selection_task or checkpoint[
                "selection_metric"
            ] != selection_metric:
                raise ValueError(
                    "resume checkpoint selection rule differs from configuration"
                )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            history = list(checkpoint["history"])
            best_score = float(checkpoint["best_score"])
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
            score = float(validation_metrics[selection_task][selection_metric])
            if not math.isfinite(score):
                raise RuntimeError("checkpoint selection metric is not finite")
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(record)
            component_summary = " ".join(
                f"{name}={train_metrics[name]:.6f}"
                for name in self.loss_component_scopes
            )
            self.logger.info(
                "epoch=%d train_loss=%.6f %s "
                "val_ctr_auc=%.6f val_cvr_auc=%.6f val_ctcvr_auc=%.6f "
                "val_ctr_gauc=%s val_cvr_gauc=%s val_ctcvr_gauc=%s "
                "val_ctr_log_loss=%.6f val_cvr_log_loss=%.6f "
                "val_ctcvr_log_loss=%.6f lr=%.6g train_seconds=%.2f "
                "val_seconds=%.2f",
                epoch,
                train_metrics["loss"],
                component_summary,
                validation_metrics["ctr"]["auc"],
                validation_metrics["cvr"]["auc"],
                validation_metrics["ctcvr"]["auc"],
                *(
                    "n/a" if metrics["gauc"] is None else f"{metrics['gauc']:.6f}"
                    for metrics in (
                        validation_metrics["ctr"],
                        validation_metrics["cvr"],
                        validation_metrics["ctcvr"],
                    )
                ),
                validation_metrics["ctr"]["log_loss"],
                validation_metrics["cvr"]["log_loss"],
                validation_metrics["ctcvr"]["log_loss"],
                learning_rate,
                train_metrics["seconds"],
                validation_metrics["seconds"],
            )

            improved = score > best_score + early_stopping_min_delta
            should_stop = False
            if improved:
                best_score = score
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                    selection_task=selection_task,
                    selection_metric=selection_metric,
                    selection_score=score,
                )
                self.logger.info(
                    "saved best checkpoint: epoch=%d %s.%s=%.6f path=%s",
                    epoch,
                    selection_task,
                    selection_metric,
                    score,
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
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "selection_task": selection_task,
                    "selection_metric": selection_metric,
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
            "selection": {
                "task": selection_task,
                "metric": selection_metric,
            },
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
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
        selection_task: str,
        selection_metric: str,
        selection_score: float,
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
                "selection_task": selection_task,
                "selection_metric": selection_metric,
                "selection_score": selection_score,
                "random_state": capture_random_state(),
            },
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
