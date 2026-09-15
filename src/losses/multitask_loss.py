"""Losses for joint CTR and traditional clicked-space CVR training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskedCVRMultiTaskLoss(nn.Module):
    """Combine exposure-space CTR BCE and clicked-space CVR BCE.

    CVR is conditional on a click. Multiplying per-example CVR loss by the
    click label prevents non-clicked impressions from being treated as
    negative post-click conversion examples.
    """

    def __init__(self, *, ctr_weight: float = 1.0, cvr_weight: float = 1.0) -> None:
        super().__init__()
        if ctr_weight <= 0.0:
            raise ValueError("ctr_weight must be positive")
        if cvr_weight <= 0.0:
            raise ValueError("cvr_weight must be positive")
        self.ctr_weight = float(ctr_weight)
        self.cvr_weight = float(cvr_weight)

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        try:
            ctr_logit = outputs["ctr_logit"]
            cvr_logit = outputs["cvr_logit"]
            click = batch["click"]
            conversion = batch["conversion"]
        except KeyError as error:
            raise ValueError(
                "outputs require ctr_logit/cvr_logit and batch requires "
                "click/conversion"
            ) from error

        tensors = (ctr_logit, cvr_logit, click, conversion)
        if not all(isinstance(tensor, Tensor) for tensor in tensors):
            raise TypeError("logits and labels must be torch tensors")
        if any(tensor.ndim != 1 for tensor in tensors):
            raise ValueError("logits and labels must all have shape [batch]")
        if not (
            ctr_logit.shape == cvr_logit.shape == click.shape == conversion.shape
        ):
            raise ValueError("logits and labels must have identical shapes")
        if not click.is_floating_point() or not conversion.is_floating_point():
            raise TypeError("click and conversion labels must be floating point")

        ctr_loss = F.binary_cross_entropy_with_logits(ctr_logit, click)
        per_example_cvr_loss = F.binary_cross_entropy_with_logits(
            cvr_logit,
            conversion,
            reduction="none",
        )
        clicked_count = click.sum()
        cvr_loss = (per_example_cvr_loss * click).sum() / clicked_count.clamp_min(1.0)
        total_loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss
        return {
            "loss": total_loss,
            "ctr_loss": ctr_loss,
            "cvr_loss": cvr_loss,
            "clicked_count": clicked_count.detach(),
        }


def _log1mexp(log_probability: Tensor) -> Tensor:
    """Compute log(1 - exp(x)) stably for x <= 0."""

    result = torch.empty_like(log_probability)
    threshold = -math.log(2.0)
    small_probability = log_probability < threshold
    result[small_probability] = torch.log1p(
        -torch.exp(log_probability[small_probability])
    )
    result[~small_probability] = torch.log(
        -torch.expm1(log_probability[~small_probability])
    )
    return result


class ESMMLoss(nn.Module):
    """Combine entire-space CTR BCE and CTCVR negative log-likelihood."""

    def __init__(
        self,
        *,
        ctr_weight: float = 1.0,
        ctcvr_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if ctr_weight <= 0.0:
            raise ValueError("ctr_weight must be positive")
        if ctcvr_weight <= 0.0:
            raise ValueError("ctcvr_weight must be positive")
        self.ctr_weight = float(ctr_weight)
        self.ctcvr_weight = float(ctcvr_weight)

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        try:
            ctr_logit = outputs["ctr_logit"]
            cvr_logit = outputs["cvr_logit"]
            click = batch["click"]
            ctcvr = batch["ctcvr"]
        except KeyError as error:
            raise ValueError(
                "outputs require ctr_logit/cvr_logit and batch requires "
                "click/ctcvr"
            ) from error

        tensors = (ctr_logit, cvr_logit, click, ctcvr)
        if not all(isinstance(tensor, Tensor) for tensor in tensors):
            raise TypeError("logits and labels must be torch tensors")
        if any(tensor.ndim != 1 for tensor in tensors):
            raise ValueError("logits and labels must all have shape [batch]")
        if not (ctr_logit.shape == cvr_logit.shape == click.shape == ctcvr.shape):
            raise ValueError("logits and labels must have identical shapes")
        if not click.is_floating_point() or not ctcvr.is_floating_point():
            raise TypeError("click and ctcvr labels must be floating point")

        # Loss arithmetic stays in FP32 even when the surrounding training
        # forward uses autocast. This matters because pCTCVR can be tiny.
        ctr_logit_float = ctr_logit.float()
        cvr_logit_float = cvr_logit.float()
        click_float = click.float()
        ctcvr_float = ctcvr.float()
        ctr_loss = F.binary_cross_entropy_with_logits(
            ctr_logit_float,
            click_float,
        )

        log_p_ctcvr = F.logsigmoid(ctr_logit_float) + F.logsigmoid(
            cvr_logit_float
        )
        log_one_minus_p_ctcvr = _log1mexp(log_p_ctcvr)
        positive_mask = ctcvr_float == 1.0
        per_example_ctcvr_loss = torch.empty_like(log_p_ctcvr)
        per_example_ctcvr_loss[positive_mask] = -log_p_ctcvr[positive_mask]
        per_example_ctcvr_loss[~positive_mask] = -log_one_minus_p_ctcvr[
            ~positive_mask
        ]
        ctcvr_loss = per_example_ctcvr_loss.mean()
        total_loss = self.ctr_weight * ctr_loss + self.ctcvr_weight * ctcvr_loss
        return {
            "loss": total_loss,
            "ctr_loss": ctr_loss,
            "ctcvr_loss": ctcvr_loss,
        }


class ESMMWithAuxiliaryCVRLoss(ESMMLoss):
    """Add funnel-aware sampled clicked-space CVR supervision to ESMM.

    CTR and CTCVR keep their original entire-space objectives.  Only the
    auxiliary CVR term samples ``click=1, conversion=0`` examples; every
    clicked conversion is retained.  Sampling is renewed on every training
    forward and uses PyTorch's checkpointed RNG state.
    """

    def __init__(
        self,
        *,
        ctr_weight: float = 1.0,
        ctcvr_weight: float = 1.0,
        auxiliary_cvr_weight: float,
        negative_to_positive_ratio: float,
    ) -> None:
        super().__init__(
            ctr_weight=ctr_weight,
            ctcvr_weight=ctcvr_weight,
        )
        if auxiliary_cvr_weight <= 0.0:
            raise ValueError("auxiliary_cvr_weight must be positive")
        if negative_to_positive_ratio <= 0.0:
            raise ValueError("negative_to_positive_ratio must be positive")
        self.auxiliary_cvr_weight = float(auxiliary_cvr_weight)
        self.negative_to_positive_ratio = float(
            negative_to_positive_ratio
        )

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        losses = super().forward(outputs, batch)
        try:
            cvr_logit = outputs["cvr_logit"]
            click = batch["click"]
            conversion = batch["conversion"]
        except KeyError as error:
            raise ValueError(
                "auxiliary CVR loss requires cvr_logit, click and conversion"
            ) from error
        if not all(
            isinstance(tensor, Tensor)
            for tensor in (cvr_logit, click, conversion)
        ):
            raise TypeError("CVR logits and labels must be torch tensors")
        if any(tensor.ndim != 1 for tensor in (cvr_logit, click, conversion)):
            raise ValueError("CVR logits and labels must have shape [batch]")
        if not (cvr_logit.shape == click.shape == conversion.shape):
            raise ValueError("CVR logits and labels must have identical shapes")
        if not click.is_floating_point() or not conversion.is_floating_point():
            raise TypeError("click and conversion labels must be floating point")

        cvr_logit_float = cvr_logit.float()
        click_mask = click.float() == 1.0
        conversion_float = conversion.float()
        positive_mask = click_mask & (conversion_float == 1.0)
        negative_mask = click_mask & (conversion_float == 0.0)

        if self.training:
            # The tensor-only probability calculation avoids a host/device
            # synchronization in every GPU batch.  Across an epoch the
            # expected sampled ratio is the configured negative:positive
            # ratio, capped at all available clicked negatives.
            positive_count = positive_mask.sum()
            negative_count = negative_mask.sum()
            negative_keep_probability = (
                self.negative_to_positive_ratio
                * positive_count.float()
                / negative_count.clamp_min(1).float()
            ).clamp(max=1.0)
            sampled_negative_mask = negative_mask & (
                torch.rand_like(cvr_logit_float)
                < negative_keep_probability
            )
            selected_mask = positive_mask | sampled_negative_mask
        else:
            selected_mask = click_mask

        per_example_loss = F.binary_cross_entropy_with_logits(
            cvr_logit_float,
            conversion_float,
            reduction="none",
        )
        selected_count = selected_mask.sum()
        auxiliary_cvr_loss = (
            per_example_loss * selected_mask.float()
        ).sum() / selected_count.clamp_min(1).float()
        sampled_positive_count = (selected_mask & positive_mask).sum()
        sampled_negative_count = (selected_mask & negative_mask).sum()
        total_loss = (
            losses["loss"]
            + self.auxiliary_cvr_weight * auxiliary_cvr_loss
        )
        return {
            **losses,
            "loss": total_loss,
            "auxiliary_cvr_loss": auxiliary_cvr_loss,
            "auxiliary_cvr_positive_count": sampled_positive_count.detach(),
            "auxiliary_cvr_negative_count": sampled_negative_count.detach(),
            "auxiliary_cvr_sampled_count": selected_count.detach(),
        }


def build_esmm_loss(loss_config: Mapping[str, Any]) -> ESMMLoss:
    """Build standard ESMM or its configured auxiliary-CVR extension."""

    ctr_weight = float(loss_config["ctr_weight"])
    ctcvr_weight = float(loss_config["ctcvr_weight"])
    auxiliary_config = loss_config.get("auxiliary_cvr_loss")
    if auxiliary_config is None:
        return ESMMLoss(
            ctr_weight=ctr_weight,
            ctcvr_weight=ctcvr_weight,
        )
    if not isinstance(auxiliary_config, Mapping):
        raise ValueError("auxiliary_cvr_loss must be a mapping")
    if not bool(auxiliary_config.get("enabled", False)):
        return ESMMLoss(
            ctr_weight=ctr_weight,
            ctcvr_weight=ctcvr_weight,
        )
    return ESMMWithAuxiliaryCVRLoss(
        ctr_weight=ctr_weight,
        ctcvr_weight=ctcvr_weight,
        auxiliary_cvr_weight=float(auxiliary_config["weight"]),
        negative_to_positive_ratio=float(
            auxiliary_config["negative_to_positive_ratio"]
        ),
    )
