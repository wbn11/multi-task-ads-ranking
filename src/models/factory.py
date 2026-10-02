"""Configuration-driven builders for saved multi-task model runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from src.data.feature_encoder import FeatureEncoder

from .dcn_esmm import DCNESMM
from .dcn_ple import DCNPLE
from .dcn_ple_esmm import DCNPLEESMM
from .esmm import ESMM
from .mmoe import MMoE
from .ple import PLE
from .shared_bottom import SharedBottom
from .target_aware_dcn_ple_esmm import TargetAwareDCNPLEESMM


def _integer_tuple(values: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(int(value) for value in values)


def build_multitask_model(
    encoder: FeatureEncoder,
    model_config: Mapping[str, Any],
) -> nn.Module:
    """Rebuild a supported multi-task model from its saved configuration."""

    model_name = str(model_config.get("name", "")).lower()
    common = {
        "embedding_dim": int(model_config["embedding_dim"]),
        "tower_hidden_dims": _integer_tuple(
            model_config["tower_hidden_dims"],
            name="tower_hidden_dims",
        ),
        "dropout": float(model_config["dropout"]),
        "embedding_pooling": str(model_config["embedding_pooling"]),
    }
    if model_name in ("shared_bottom", "esmm"):
        model_class = SharedBottom if model_name == "shared_bottom" else ESMM
        return model_class.from_feature_encoder(
            encoder,
            shared_hidden_dims=_integer_tuple(
                model_config["shared_hidden_dims"],
                name="shared_hidden_dims",
            ),
            **common,
        )
    if model_name == "dcn_esmm":
        return DCNESMM.from_feature_encoder(
            encoder,
            num_cross_layers=int(model_config["num_cross_layers"]),
            cross_layer_norm=bool(model_config.get("cross_layer_norm", True)),
            shared_hidden_dims=_integer_tuple(
                model_config["shared_hidden_dims"],
                name="shared_hidden_dims",
            ),
            **common,
        )
    if model_name == "mmoe":
        return MMoE.from_feature_encoder(
            encoder,
            num_experts=int(model_config["num_experts"]),
            expert_hidden_dims=_integer_tuple(
                model_config["expert_hidden_dims"],
                name="expert_hidden_dims",
            ),
            gate_dropout=float(model_config.get("gate_dropout", 0.0)),
            **common,
        )
    if model_name in ("ple", "ple_esmm"):
        return PLE.from_feature_encoder(
            encoder,
            num_shared_experts=int(model_config["num_shared_experts"]),
            num_task_experts=int(model_config["num_task_experts"]),
            expert_hidden_dims=_integer_tuple(
                model_config["expert_hidden_dims"],
                name="expert_hidden_dims",
            ),
            gate_dropout=float(model_config.get("gate_dropout", 0.0)),
            **common,
        )
    if model_name in ("dcn_ple", "dcn_ple_esmm"):
        model_class = DCNPLE if model_name == "dcn_ple" else DCNPLEESMM
        return model_class.from_feature_encoder(
            encoder,
            num_cross_layers=int(model_config["num_cross_layers"]),
            cross_layer_norm=bool(model_config.get("cross_layer_norm", True)),
            num_shared_experts=int(model_config["num_shared_experts"]),
            num_task_experts=int(model_config["num_task_experts"]),
            expert_hidden_dims=_integer_tuple(
                model_config["expert_hidden_dims"],
                name="expert_hidden_dims",
            ),
            gate_dropout=float(model_config.get("gate_dropout", 0.0)),
            **common,
        )
    if model_name == "target_aware_dcn_ple_esmm":
        return TargetAwareDCNPLEESMM.from_feature_encoder(
            encoder,
            num_cross_layers=int(model_config["num_cross_layers"]),
            cross_layer_norm=bool(model_config.get("cross_layer_norm", True)),
            num_shared_experts=int(model_config["num_shared_experts"]),
            num_task_experts=int(model_config["num_task_experts"]),
            expert_hidden_dims=_integer_tuple(
                model_config["expert_hidden_dims"],
                name="expert_hidden_dims",
            ),
            gate_dropout=float(model_config.get("gate_dropout", 0.0)),
            history_attention_hidden_dim=int(
                model_config["history_attention_hidden_dim"]
            ),
            history_count_prior_strength=float(
                model_config.get("history_count_prior_strength", 1.0)
            ),
            **common,
        )
    raise ValueError(f"unsupported multi-task model: {model_name!r}")
