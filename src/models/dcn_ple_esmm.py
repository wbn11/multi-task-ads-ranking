"""DCNv2-enhanced PLE model trained with the ESMM objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from src.data.feature_encoder import FeatureEncoder
from src.layers.cross_network import CrossNetworkV2
from src.models.ple import PLE


class DCNPLEESMM(PLE):
    """Feed explicit DCNv2 crosses into task-separated PLE experts.

    The network emits CTR and CVR heads and preserves the funnel identity
    ``pCTCVR = pCTR * pCVR``. The ESMM entire-space objective is supplied by
    :class:`src.trainer.dcn_ple_esmm_trainer.DCNPLEESMMTrainer` so the model
    remains usable for inference without coupling it to a loss function.
    """

    def __init__(
        self,
        field_vocab_sizes: Mapping[str, int],
        *,
        embedding_dim: int,
        num_cross_layers: int,
        cross_layer_norm: bool,
        num_shared_experts: int,
        num_task_experts: int,
        expert_hidden_dims: Sequence[int],
        tower_hidden_dims: Sequence[int],
        dropout: float,
        gate_dropout: float = 0.0,
        embedding_pooling: str = "weighted_mean",
    ) -> None:
        super().__init__(
            field_vocab_sizes,
            embedding_dim=embedding_dim,
            num_shared_experts=num_shared_experts,
            num_task_experts=num_task_experts,
            expert_hidden_dims=expert_hidden_dims,
            tower_hidden_dims=tower_hidden_dims,
            dropout=dropout,
            gate_dropout=gate_dropout,
            embedding_pooling=embedding_pooling,
        )
        self.num_cross_layers = int(num_cross_layers)
        self.cross_layer_norm_enabled = bool(cross_layer_norm)
        self.cross_network = CrossNetworkV2(
            self.input_dim,
            self.num_cross_layers,
        )
        self.cross_normalization: nn.Module = (
            nn.LayerNorm(self.input_dim)
            if self.cross_layer_norm_enabled
            else nn.Identity()
        )

    @classmethod
    def from_feature_encoder(
        cls,
        encoder: FeatureEncoder,
        *,
        embedding_dim: int,
        num_cross_layers: int,
        cross_layer_norm: bool,
        num_shared_experts: int,
        num_task_experts: int,
        expert_hidden_dims: Sequence[int],
        tower_hidden_dims: Sequence[int],
        dropout: float,
        gate_dropout: float = 0.0,
        embedding_pooling: str = "weighted_mean",
    ) -> "DCNPLEESMM":
        return cls(
            {
                field_id: vocabulary.vocab_size
                for field_id, vocabulary in encoder.field_vocabularies.items()
            },
            embedding_dim=embedding_dim,
            num_cross_layers=num_cross_layers,
            cross_layer_norm=cross_layer_norm,
            num_shared_experts=num_shared_experts,
            num_task_experts=num_task_experts,
            expert_hidden_dims=expert_hidden_dims,
            tower_hidden_dims=tower_hidden_dims,
            dropout=dropout,
            gate_dropout=gate_dropout,
            embedding_pooling=embedding_pooling,
        )

    def _model_input(self, batch: Mapping[str, Any]) -> Tensor:
        raw_input = super()._model_input(batch)
        crossed = self.cross_network(raw_input)
        return self.cross_normalization(crossed)
