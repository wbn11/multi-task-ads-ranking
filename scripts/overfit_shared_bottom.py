"""Overfit one label-balanced Ali-CCP batch to validate Shared Bottom."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch import Tensor, nn

from src.data.dataset import AdsDataset, collate_ads_batch
from src.data.debug_batch import select_funnel_overfit_samples
from src.data.feature_encoder import FeatureEncoder
from src.losses.multitask_loss import MaskedCVRMultiTaskLoss
from src.models.shared_bottom import SharedBottom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/shared_bottom.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
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
    model: SharedBottom,
    criterion: MaskedCVRMultiTaskLoss,
    batch: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Tensor]]:
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
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


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(args.config))
    model_config = config["model"]
    if str(model_config["name"]).lower() != "shared_bottom":
        raise ValueError(
            "overfit_shared_bottom.py requires model.name=shared_bottom"
        )
    loss_config = config["loss"]
    overfit_config = config["overfit"]

    data_config = load_yaml(project_path(config["data"]["config"]))["data"]
    processed_dir = project_path(data_config["processed_dir"])
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
        batch = move_batch(collate_ads_batch(samples), device)
        click = batch["click"]
        conversion = batch["conversion"]
        click_positive_mask = click == 1.0
        click_negative_mask = click == 0.0
        conversion_positive_mask = click_positive_mask & (conversion == 1.0)
        conversion_negative_mask = click_positive_mask & (conversion == 0.0)

        encoder = FeatureEncoder.load(processed_dir / "vocab.json")
        model = SharedBottom.from_feature_encoder(
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
                        "cvr_loss": float(losses["cvr_loss"].item()),
                    }
                )

        final_losses, final_outputs = evaluate_fixed_batch(model, criterion, batch)
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
        valid = bool(
            all(math.isfinite(value) for value in final_losses.values())
            and gradients_are_finite
            and all(
                reductions[name] >= minimum_loss_reduction
                for name in ("loss", "ctr_loss", "cvr_loss")
            )
            and ctr_positive_mean > ctr_negative_mean
            and cvr_positive_mean > cvr_negative_mean
            and ctcvr_product_max_error <= 1e-7
        )
        report = {
            "valid": valid,
            "model": "shared_bottom",
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
            "shared_hidden_dims": list(model_config["shared_hidden_dims"]),
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
