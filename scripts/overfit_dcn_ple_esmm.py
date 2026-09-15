"""Overfit one balanced batch to validate the DCN-PLE-ESMM model family."""

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
from src.losses.multitask_loss import (
    ESMMLoss,
    ESMMWithAuxiliaryCVRLoss,
    build_esmm_loss,
)
from src.models.dcn_ple_esmm import DCNPLEESMM
from src.models.factory import build_multitask_model
from src.trainer.ctr_experiment import load_yaml, project_path, resolve_device, set_seed
from src.trainer.ctr_trainer_base import move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dcn_ple_esmm.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def _received_finite_nonzero_gradient(module: nn.Module) -> bool:
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


def _all_gradients_finite(model: nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return bool(
        gradients
        and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    )


def _mean(probabilities: Tensor, mask: Tensor) -> float:
    selected = probabilities[mask]
    return float(selected.mean().item()) if selected.numel() else math.nan


def _evaluate(
    model: DCNPLEESMM,
    criterion: ESMMLoss,
    batch: Mapping[str, Any],
    loss_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, Tensor]]:
    model.eval()
    criterion.eval()
    with torch.no_grad():
        outputs = model(batch, return_gate_weights=True)
        losses = criterion(outputs, batch)
    return (
        {
            name: float(losses[name].item())
            for name in loss_names
        },
        outputs,
    )


def _build_model(
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
    model_config = config["model"]
    model_name = str(model_config["name"]).lower()
    supported_models = ("dcn_ple_esmm", "target_aware_dcn_ple_esmm")
    if model_name not in supported_models:
        raise ValueError(
            "overfit_dcn_ple_esmm.py requires model.name in "
            f"{supported_models}; received {model_name!r}"
        )
    loss_config = config["loss"]
    overfit_config = config["overfit"]
    data_file = load_yaml(
        project_path(PROJECT_ROOT, config["data"]["config"])
    )
    processed_dir = project_path(
        PROJECT_ROOT,
        data_file["data"]["processed_dir"],
    )
    split = str(config["data"].get("split", "train"))
    batch_size = int(
        overfit_config["batch_size"]
        if args.batch_size is None
        else args.batch_size
    )
    steps = int(overfit_config["steps"] if args.steps is None else args.steps)
    seed = int(overfit_config["seed"])
    minimum_reduction = float(overfit_config["minimum_loss_reduction"])
    gate_tolerance = float(overfit_config["gate_sum_tolerance"])
    if steps <= 0 or batch_size < 3:
        raise ValueError("steps must be positive and batch_size at least three")

    set_seed(seed)
    device = resolve_device(str(args.device or overfit_config["device"]))
    dataset = AdsDataset(processed_dir, split=split)
    try:
        samples = select_funnel_overfit_samples(
            dataset,
            batch_size=batch_size,
            conversion_positives=int(overfit_config["conversion_positives"]),
            clicked_non_conversions=int(
                overfit_config["clicked_non_conversions"]
            ),
            seed=seed,
        )
        batch = move_batch_to_device(collate_ads_batch(samples), device)
        click = batch["click"]
        conversion = batch["conversion"]
        ctr_positive = click == 1.0
        ctr_negative = click == 0.0
        cvr_positive = ctr_positive & (conversion == 1.0)
        cvr_negative = ctr_positive & (conversion == 0.0)

        encoder = FeatureEncoder.load(processed_dir / "vocab.json")
        model = _build_model(encoder, model_config).to(device)
        criterion = build_esmm_loss(loss_config)
        auxiliary_cvr_enabled = isinstance(
            criterion,
            ESMMWithAuxiliaryCVRLoss,
        )
        loss_names = ("loss", "ctr_loss", "ctcvr_loss")
        if auxiliary_cvr_enabled:
            loss_names += ("auxiliary_cvr_loss",)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(overfit_config["learning_rate"]),
            weight_decay=float(overfit_config["weight_decay"]),
        )

        initial_losses, _ = _evaluate(model, criterion, batch, loss_names)
        trace_steps = {1, steps, *(max(1, steps * part // 4) for part in range(1, 4))}
        loss_trace: list[dict[str, float | int]] = []
        gradient_checks = {
            "cross_network": False,
            "shared_experts": False,
            "ctr_experts": False,
            "cvr_experts": False,
            "ctr_gate": False,
            "cvr_gate": False,
        }
        if hasattr(model, "history_attention"):
            gradient_checks["history_attention"] = False
        gradients_finite = True
        auxiliary_count_names = (
            "auxiliary_cvr_positive_count",
            "auxiliary_cvr_negative_count",
            "auxiliary_cvr_sampled_count",
        )
        auxiliary_counts = {name: 0 for name in auxiliary_count_names}
        model.train()
        criterion.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = criterion(outputs, batch)
            if auxiliary_cvr_enabled:
                for name in auxiliary_count_names:
                    auxiliary_counts[name] += int(losses[name].item())
            loss = losses["loss"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            gradients_finite = gradients_finite and _all_gradients_finite(model)
            if not gradients_finite:
                raise RuntimeError(f"non-finite gradient at step {step}")
            modules = {
                "cross_network": model.cross_network,
                "shared_experts": model.shared_experts,
                "ctr_experts": model.ctr_experts,
                "cvr_experts": model.cvr_experts,
                "ctr_gate": model.ctr_gate,
                "cvr_gate": model.cvr_gate,
            }
            if hasattr(model, "history_attention"):
                modules["history_attention"] = model.history_attention
            gradient_checks = {
                name: gradient_checks[name]
                or _received_finite_nonzero_gradient(module)
                for name, module in modules.items()
            }
            optimizer.step()
            if step in trace_steps:
                trace_record: dict[str, float | int] = {
                    "step": step,
                    **{
                        name: float(losses[name].item())
                        for name in loss_names
                    },
                }
                if auxiliary_cvr_enabled:
                    trace_record.update(
                        {
                            name: int(losses[name].item())
                            for name in auxiliary_count_names
                        }
                    )
                loss_trace.append(trace_record)

        final_losses, final_outputs = _evaluate(
            model,
            criterion,
            batch,
            loss_names,
        )
        reductions = {
            name: (initial_losses[name] - final_losses[name]) / initial_losses[name]
            for name in initial_losses
        }
        prediction_means = {
            "ctr_positive": _mean(final_outputs["ctr"], ctr_positive),
            "ctr_negative": _mean(final_outputs["ctr"], ctr_negative),
            "cvr_positive": _mean(final_outputs["cvr"], cvr_positive),
            "cvr_negative": _mean(final_outputs["cvr"], cvr_negative),
        }
        gate_errors = {
            task: float(
                (
                    final_outputs[f"{task}_gate_weights"].sum(dim=1) - 1.0
                ).abs().max().item()
            )
            for task in ("ctr", "cvr")
        }
        product_error = float(
            (
                final_outputs["ctcvr"]
                - final_outputs["ctr"] * final_outputs["cvr"]
            ).abs().max().item()
        )
        outputs_finite = all(
            bool(torch.isfinite(tensor).all())
            for tensor in final_outputs.values()
        )
        valid = bool(
            outputs_finite
            and gradients_finite
            and all(gradient_checks.values())
            and all(value >= minimum_reduction for value in reductions.values())
            and prediction_means["ctr_positive"]
            > prediction_means["ctr_negative"]
            and prediction_means["cvr_positive"]
            > prediction_means["cvr_negative"]
            and product_error <= 1e-7
            and all(value <= gate_tolerance for value in gate_errors.values())
        )
        report = {
            "valid": valid,
            "model": model_name,
            "device": str(device),
            "split": split,
            "dataset_size": len(dataset),
            "batch_size": int(click.numel()),
            "field_count": len(model.field_ids),
            "embedding_dim": model.embedding_dim,
            "num_cross_layers": model.num_cross_layers,
            "cross_layer_norm": model.cross_layer_norm_enabled,
            "num_shared_experts": model.num_shared_experts,
            "num_task_experts": model.num_task_experts,
            "gate_dropout": model.gate_dropout,
            "auxiliary_cvr_sampling": (
                {
                    "enabled": True,
                    "loss_weight": criterion.auxiliary_cvr_weight,
                    "requested_negative_to_positive_ratio": (
                        criterion.negative_to_positive_ratio
                    ),
                    "sampled_positive_count_during_training": (
                        auxiliary_counts["auxiliary_cvr_positive_count"]
                    ),
                    "sampled_negative_count_during_training": (
                        auxiliary_counts["auxiliary_cvr_negative_count"]
                    ),
                    "sampled_count_during_training": (
                        auxiliary_counts["auxiliary_cvr_sampled_count"]
                    ),
                    "observed_negative_to_positive_ratio": (
                        auxiliary_counts["auxiliary_cvr_negative_count"]
                        / max(
                            auxiliary_counts[
                                "auxiliary_cvr_positive_count"
                            ],
                            1,
                        )
                    ),
                }
                if auxiliary_cvr_enabled
                else {"enabled": False}
            ),
            "history_attention": (
                {
                    "field_pairs": [
                        list(pair) for pair in model.history_target_field_pairs
                    ],
                    "hidden_dim": model.history_attention_hidden_dim,
                    "count_prior_strength": (
                        model.history_count_prior_strength
                    ),
                }
                if hasattr(model, "history_attention")
                else None
            ),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "steps": steps,
            "initial_losses": initial_losses,
            "final_losses": final_losses,
            "loss_reduction_fractions": reductions,
            "minimum_loss_reduction": minimum_reduction,
            "prediction_means": prediction_means,
            "ctcvr_product_max_error": product_error,
            "gate_sum_max_errors": gate_errors,
            "gate_sum_tolerance": gate_tolerance,
            "gradient_coverage": gradient_checks,
            "outputs_finite": outputs_finite,
            "gradients_finite": gradients_finite,
            "loss_trace": loss_trace,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not valid:
            raise SystemExit(1)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
