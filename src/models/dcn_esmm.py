"""DCNv2-enhanced Shared Bottom trained with the ESMM objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from src.data.feature_encoder import FeatureEncoder
from src.layers.cross_network import CrossNetworkV2

from .esmm import ESMM


class DCNESMM(ESMM):
    """Feed explicit DCNv2 crosses into an ESMM Shared Bottom backbone.

    This model is the no-PLE cell in the DCNv2 x PLE factorial ablation.  It
    keeps the same CTR/CVR heads and ``pCTCVR = pCTR * pCVR`` identity as ESMM;
    the trainer supplies the entire-space CTR + CTCVR objective.
    """

    def __init__(
        self,
        field_vocab_sizes: Mapping[str, int],
        *,
        embedding_dim: int,
        num_cross_layers: int,
        cross_layer_norm: bool,
        shared_hidden_dims: Sequence[int],
        tower_hidden_dims: Sequence[int],
        dropout: float,
        embedding_pooling: str = "weighted_mean",
    ) -> None:
        super().__init__(
            field_vocab_sizes,
            embedding_dim=embedding_dim,
            shared_hidden_dims=shared_hidden_dims,
            tower_hidden_dims=tower_hidden_dims,
            dropout=dropout,
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
        shared_hidden_dims: Sequence[int],
        tower_hidden_dims: Sequence[int],
        dropout: float,
        embedding_pooling: str = "weighted_mean",
    ) -> "DCNESMM":
        return cls(
            {
                field_id: vocabulary.vocab_size
                for field_id, vocabulary in encoder.field_vocabularies.items()
            },
            embedding_dim=embedding_dim,
            num_cross_layers=num_cross_layers,
            cross_layer_norm=cross_layer_norm,
            shared_hidden_dims=shared_hidden_dims,
            tower_hidden_dims=tower_hidden_dims,
            dropout=dropout,
            embedding_pooling=embedding_pooling,
        )

    def _model_input(self, batch: Mapping[str, Any]) -> Tensor:
        raw_input = super()._model_input(batch)
        crossed = self.cross_network(raw_input)
        return self.cross_normalization(crossed)
