"""Overfit one label-balanced Ali-CCP batch to validate single-level PLE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor, nn

from src.data.dataset import AdsDataset, collate_ads_batch
from src.data.debug_batch import select_funnel_overfit_samples
from src.data.feature_encoder import FeatureEncoder
from src.losses.multitask_loss import MaskedCVRMultiTaskLoss
from src.models.ple import PLE
from src.trainer.ctr_experiment import (
    load_yaml,
    project_path,
    resolve_device,
    set_seed,
)
from src.trainer.ctr_trainer_base import move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ple.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def finite_gradients(model: nn.Module) -> bool:
    checks = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return bool(checks and torch.stack(checks).all())


def received_nonzero_gradient(module: nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return bool(
        gradients
        and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        and any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    )


def mean_for_mask(probabilities: Tensor, mask: Tensor) -> float:
    selected = probabilities[mask]
    return float(selected.mean().item()) if selected.numel() else math.nan


def evaluate_fixed_batch(
    model: PLE,
    criterion: MaskedCVRMultiTaskLoss,
    batch: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Tensor]]:
    model.eval()
    with torch.no_grad():
        outputs = model(batch, return_gate_weights=True)
        losses = criterion(outputs, batch)
    return (
        {
            name: float(losses[name].item())
            for name in ("loss", "ctr_loss", "cvr_loss")
        },
        outputs,
    )


def loss_reduction(initial: float, final: float) -> float:
    return (initial - final) / initial


def update_coverage(
    current: list[bool],
    modules: nn.ModuleList,
) -> list[bool]:
    return [
        covered or received_nonzero_gradient(module)
        for covered, module in zip(current, modules, strict=True)
    ]


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(PROJECT_ROOT, args.config))
    model_config = config["model"]
    if str(model_config["name"]).lower() != "ple":
        raise ValueError("overfit_ple.py requires model.name=ple")
    loss_config = config["loss"]
    overfit_config = config["overfit"]

    data_config_path = project_path(PROJECT_ROOT, config["data"]["config"])
    data_config = load_yaml(data_config_path)["data"]
    processed_dir = project_path(PROJECT_ROOT, data_config["processed_dir"])
    split = str(config["data"].get("split", "train"))
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else overfit_config["batch_size"]
    )
    steps = int(args.steps if args.steps is not None else overfit_config["steps"])
    requested_device = str(args.device or overfit_config["device"])
    seed = int(overfit_config["seed"])
    minimum_loss_reduction = float(overfit_config["minimum_loss_reduction"])
    gate_sum_tolerance = float(overfit_config["gate_sum_tolerance"])
    conversion_positives = int(overfit_config["conversion_positives"])
    clicked_non_conversions = int(overfit_config["clicked_non_conversions"])

    if batch_size < 3:
        raise ValueError("batch_size must be at least three")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < minimum_loss_reduction < 1.0:
        raise ValueError("minimum_loss_reduction must be between zero and one")
    if gate_sum_tolerance <= 0.0:
        raise ValueError("gate_sum_tolerance must be positive")

    set_seed(seed)
    device = resolve_device(requested_device)
    dataset = AdsDataset(processed_dir, split=split)
    try:
        samples = select_funnel_overfit_samples(
            dataset,
            batch_size=batch_size,
            conversion_positives=conversion_positives,
            clicked_non_conversions=clicked_non_conversions,
            seed=seed,
        )
        batch = move_batch_to_device(collate_ads_batch(samples), device)
        click = batch["click"]
        conversion = batch["conversion"]
        click_positive_mask = click == 1.0
        click_negative_mask = click == 0.0
        conversion_positive_mask = click_positive_mask & (conversion == 1.0)
        conversion_negative_mask = click_positive_mask & (conversion == 0.0)

        encoder = FeatureEncoder.load(processed_dir / "vocab.json")
        model = PLE.from_feature_encoder(
            encoder,
            embedding_dim=int(model_config["embedding_dim"]),
            num_shared_experts=int(model_config["num_shared_experts"]),
            num_task_experts=int(model_config["num_task_experts"]),
            expert_hidden_dims=tuple(
                int(value) for value in model_config["expert_hidden_dims"]
            ),
            tower_hidden_dims=tuple(
                int(value) for value in model_config["tower_hidden_dims"]
            ),
            dropout=float(model_config["dropout"]),
            gate_dropout=float(model_config["gate_dropout"]),
            embedding_pooling=str(model_config["embedding_pooling"]),
        ).to(device)
        criterion = MaskedCVRMultiTaskLoss(
            ctr_weight=float(loss_config["ctr_weight"]),
            cvr_weight=float(loss_config["cvr_weight"]),
        )
        learning_rate = float(overfit_config["learning_rate"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(overfit_config["weight_decay"]),
        )

        initial_losses, _ = evaluate_fixed_batch(model, criterion, batch)
        trace_steps = {1, steps}
        trace_steps.update(
            range(max(1, steps // 4), steps + 1, max(1, steps // 4))
        )
        loss_trace: list[dict[str, float | int]] = []
        gradients_are_finite = True
        shared_coverage = [False] * model.num_shared_experts
        ctr_coverage = [False] * model.num_task_experts
        cvr_coverage = [False] * model.num_task_experts
        ctr_gate_received_gradient = False
        cvr_gate_received_gradient = False
        model.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = criterion(outputs, batch)
            loss = losses["loss"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            gradients_are_finite = gradients_are_finite and finite_gradients(model)
            if not gradients_are_finite:
                raise RuntimeError(f"non-finite gradient at step {step}")
            shared_coverage = update_coverage(
                shared_coverage,
                model.shared_experts,
            )
            ctr_coverage = update_coverage(ctr_coverage, model.ctr_experts)
            cvr_coverage = update_coverage(cvr_coverage, model.cvr_experts)
            ctr_gate_received_gradient = (
                ctr_gate_received_gradient
                or received_nonzero_gradient(model.ctr_gate)
            )
            cvr_gate_received_gradient = (
                cvr_gate_received_gradient
                or received_nonzero_gradient(model.cvr_gate)
            )
            optimizer.step()
            if step in trace_steps:
                loss_trace.append(
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "ctr_loss": float(losses["ctr_loss"].item()),
                        "cvr_loss": float(losses["cvr_loss"].item()),
                    }
                )

        final_losses, final_outputs = evaluate_fixed_batch(
            model,
            criterion,
            batch,
        )
        reductions = {
            name: loss_reduction(initial_losses[name], final_losses[name])
            for name in ("loss", "ctr_loss", "cvr_loss")
        }
        ctr_positive_mean = mean_for_mask(
            final_outputs["ctr"], click_positive_mask
        )
        ctr_negative_mean = mean_for_mask(
            final_outputs["ctr"], click_negative_mask
        )
        cvr_positive_mean = mean_for_mask(
            final_outputs["cvr"], conversion_positive_mask
        )
        cvr_negative_mean = mean_for_mask(
            final_outputs["cvr"], conversion_negative_mask
        )
        ctcvr_product_max_error = float(
            (
                final_outputs["ctcvr"]
                - final_outputs["ctr"] * final_outputs["cvr"]
            )
            .abs()
            .max()
            .item()
        )
        ctr_gate_weights = final_outputs["ctr_gate_weights"]
        cvr_gate_weights = final_outputs["cvr_gate_weights"]
        ctr_gate_sum_max_error = float(
            (ctr_gate_weights.sum(dim=1) - 1.0).abs().max().item()
        )
        cvr_gate_sum_max_error = float(
            (cvr_gate_weights.sum(dim=1) - 1.0).abs().max().item()
        )
        outputs_are_finite = all(
            bool(torch.isfinite(final_outputs[name]).all())
            for name in (
                "ctr_logit",
                "cvr_logit",
                "ctr",
                "cvr",
                "ctcvr",
                "ctr_gate_weights",
                "cvr_gate_weights",
            )
        )
        valid = bool(
            all(math.isfinite(value) for value in final_losses.values())
            and outputs_are_finite
            and gradients_are_finite
            and all(shared_coverage)
            and all(ctr_coverage)
            and all(cvr_coverage)
            and ctr_gate_received_gradient
            and cvr_gate_received_gradient
            and all(
                reductions[name] >= minimum_loss_reduction
                for name in ("loss", "ctr_loss", "cvr_loss")
            )
            and ctr_positive_mean > ctr_negative_mean
            and cvr_positive_mean > cvr_negative_mean
            and ctcvr_product_max_error <= 1e-7
            and ctr_gate_sum_max_error <= gate_sum_tolerance
            and cvr_gate_sum_max_error <= gate_sum_tolerance
        )
        report = {
            "valid": valid,
            "model": "ple",
            "targets": ["click", "conversion"],
            "device": str(device),
            "split": split,
            "dataset_size": len(dataset),
            "batch_size": int(click.numel()),
            "click_positive_labels": int(click_positive_mask.sum().item()),
            "click_negative_labels": int(click_negative_mask.sum().item()),
            "conversion_positive_labels": int(
                conversion_positive_mask.sum().item()
            ),
            "clicked_non_conversion_labels": int(
                conversion_negative_mask.sum().item()
            ),
            "field_count": len(model.field_ids),
            "embedding_dim": model.embedding_dim,
            "embedding_pooling": model.embedding_pooling,
            "num_shared_experts": model.num_shared_experts,
            "num_task_experts": model.num_task_experts,
            "total_experts": (
                model.num_shared_experts + 2 * model.num_task_experts
            ),
            "experts_per_task": model.experts_per_task,
            "gate_expert_order": {
                "ctr": [
                    *(
                        f"shared_{index + 1}"
                        for index in range(model.num_shared_experts)
                    ),
                    *(
                        f"ctr_specific_{index + 1}"
                        for index in range(model.num_task_experts)
                    ),
                ],
                "cvr": [
                    *(
                        f"shared_{index + 1}"
                        for index in range(model.num_shared_experts)
                    ),
                    *(
                        f"cvr_specific_{index + 1}"
                        for index in range(model.num_task_experts)
                    ),
                ],
            },
            "gate_dropout": model.gate_dropout,
            "expert_hidden_dims": list(model_config["expert_hidden_dims"]),
            "tower_hidden_dims": list(model_config["tower_hidden_dims"]),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "steps": steps,
            "learning_rate": learning_rate,
            "loss_weights": {
                "ctr": criterion.ctr_weight,
                "cvr": criterion.cvr_weight,
            },
            "initial_losses": initial_losses,
            "final_losses": final_losses,
            "loss_reduction_fractions": reductions,
            "minimum_loss_reduction": minimum_loss_reduction,
            "ctr_positive_prediction_mean": ctr_positive_mean,
            "ctr_negative_prediction_mean": ctr_negative_mean,
            "cvr_positive_prediction_mean": cvr_positive_mean,
            "cvr_negative_prediction_mean": cvr_negative_mean,
            "ctcvr_product_max_error": ctcvr_product_max_error,
            "ctr_gate_mean_weights": [
                float(value) for value in ctr_gate_weights.mean(dim=0).tolist()
            ],
            "cvr_gate_mean_weights": [
                float(value) for value in cvr_gate_weights.mean(dim=0).tolist()
            ],
            "ctr_gate_sum_max_error": ctr_gate_sum_max_error,
            "cvr_gate_sum_max_error": cvr_gate_sum_max_error,
            "gate_sum_tolerance": gate_sum_tolerance,
            "shared_expert_gradient_coverage": shared_coverage,
            "ctr_expert_gradient_coverage": ctr_coverage,
            "cvr_expert_gradient_coverage": cvr_coverage,
            "ctr_gate_received_gradient": ctr_gate_received_gradient,
            "cvr_gate_received_gradient": cvr_gate_received_gradient,
            "outputs_finite": outputs_are_finite,
            "gradients_finite": gradients_are_finite,
            "loss_trace": loss_trace,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not valid:
            raise SystemExit(1)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
