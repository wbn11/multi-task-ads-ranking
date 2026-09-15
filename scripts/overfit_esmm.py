"""Overfit one label-balanced Ali-CCP batch to validate ESMM."""

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
from src.losses.multitask_loss import ESMMLoss
from src.models.esmm import ESMM
from src.trainer.ctr_experiment import (
    load_yaml,
    project_path,
    resolve_device,
    set_seed,
)
from src.trainer.ctr_trainer_base import move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/esmm.yaml")
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


def mean_for_mask(probabilities: Tensor, mask: Tensor) -> float:
    selected = probabilities[mask]
    return float(selected.mean().item()) if selected.numel() else math.nan


def evaluate_fixed_batch(
    model: ESMM,
    criterion: ESMMLoss,
    batch: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Tensor]]:
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        losses = criterion(outputs, batch)
    return (
        {
            name: float(losses[name].item())
            for name in ("loss", "ctr_loss", "ctcvr_loss")
        },
        outputs,
    )


def loss_reduction(initial: float, final: float) -> float:
    return (initial - final) / initial


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(PROJECT_ROOT, args.config))
    model_config = config["model"]
    if str(model_config["name"]).lower() != "esmm":
        raise ValueError("overfit_esmm.py requires model.name=esmm")
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
    conversion_positives = int(overfit_config["conversion_positives"])
    clicked_non_conversions = int(overfit_config["clicked_non_conversions"])

    if batch_size < 3:
        raise ValueError("batch_size must be at least three")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < minimum_loss_reduction < 1.0:
        raise ValueError("minimum_loss_reduction must be between zero and one")

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
        ctcvr_label = batch["ctcvr"]
        click_positive_mask = click == 1.0
        click_negative_mask = click == 0.0
        ctcvr_positive_mask = ctcvr_label == 1.0
        ctcvr_negative_mask = ctcvr_label == 0.0
        clicked_non_conversion_mask = click_positive_mask & ctcvr_negative_mask

        encoder = FeatureEncoder.load(processed_dir / "vocab.json")
        model = ESMM.from_feature_encoder(
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
        ).to(device)
        criterion = ESMMLoss(
            ctr_weight=float(loss_config["ctr_weight"]),
            ctcvr_weight=float(loss_config["ctcvr_weight"]),
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
            optimizer.step()
            if step in trace_steps:
                loss_trace.append(
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "ctr_loss": float(losses["ctr_loss"].item()),
                        "ctcvr_loss": float(losses["ctcvr_loss"].item()),
                    }
                )

        final_losses, final_outputs = evaluate_fixed_batch(model, criterion, batch)
        reductions = {
            name: loss_reduction(initial_losses[name], final_losses[name])
            for name in ("loss", "ctr_loss", "ctcvr_loss")
        }
        ctr_positive_mean = mean_for_mask(
            final_outputs["ctr"], click_positive_mask
        )
        ctr_negative_mean = mean_for_mask(
            final_outputs["ctr"], click_negative_mask
        )
        cvr_positive_mean = mean_for_mask(
            final_outputs["cvr"], ctcvr_positive_mask
        )
        cvr_negative_mean = mean_for_mask(
            final_outputs["cvr"], clicked_non_conversion_mask
        )
        ctcvr_positive_mean = mean_for_mask(
            final_outputs["ctcvr"], ctcvr_positive_mask
        )
        ctcvr_negative_mean = mean_for_mask(
            final_outputs["ctcvr"], ctcvr_negative_mask
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
        outputs_are_finite = all(
            bool(torch.isfinite(final_outputs[name]).all())
            for name in ("ctr", "cvr", "ctcvr")
        )
        valid = bool(
            all(math.isfinite(value) for value in final_losses.values())
            and outputs_are_finite
            and gradients_are_finite
            and all(
                reductions[name] >= minimum_loss_reduction
                for name in ("loss", "ctr_loss", "ctcvr_loss")
            )
            and ctr_positive_mean > ctr_negative_mean
            and cvr_positive_mean > cvr_negative_mean
            and ctcvr_positive_mean > ctcvr_negative_mean
            and ctcvr_product_max_error <= 1e-7
        )
        report = {
            "valid": valid,
            "model": "esmm",
            "targets": ["click", "conversion"],
            "device": str(device),
            "split": split,
            "dataset_size": len(dataset),
            "batch_size": int(click.numel()),
            "click_positive_labels": int(click_positive_mask.sum().item()),
            "click_negative_labels": int(click_negative_mask.sum().item()),
            "ctcvr_positive_labels": int(ctcvr_positive_mask.sum().item()),
            "ctcvr_negative_labels": int(ctcvr_negative_mask.sum().item()),
            "clicked_non_conversion_labels": int(
                clicked_non_conversion_mask.sum().item()
            ),
            "field_count": len(model.field_ids),
            "embedding_dim": model.embedding_dim,
            "embedding_pooling": model.embedding_pooling,
            "shared_hidden_dims": list(model_config["shared_hidden_dims"]),
            "tower_hidden_dims": list(model_config["tower_hidden_dims"]),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "steps": steps,
            "learning_rate": learning_rate,
            "loss_weights": {
                "ctr": criterion.ctr_weight,
                "ctcvr": criterion.ctcvr_weight,
            },
            "initial_losses": initial_losses,
            "final_losses": final_losses,
            "loss_reduction_fractions": reductions,
            "minimum_loss_reduction": minimum_loss_reduction,
            "ctr_positive_prediction_mean": ctr_positive_mean,
            "ctr_negative_prediction_mean": ctr_negative_mean,
            "cvr_positive_prediction_mean": cvr_positive_mean,
            "cvr_clicked_non_conversion_prediction_mean": cvr_negative_mean,
            "ctcvr_positive_prediction_mean": ctcvr_positive_mean,
            "ctcvr_negative_prediction_mean": ctcvr_negative_mean,
            "ctcvr_product_max_error": ctcvr_product_max_error,
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
