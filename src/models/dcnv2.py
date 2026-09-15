"""DCNv2 CTR model for ragged Ali-CCP categorical features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from src.data.feature_encoder import FeatureEncoder
from src.data.feature_schema import ALL_FIELD_IDS
from src.layers.cross_network import CrossNetworkV2
from src.layers.embedding import SparseFeatureEmbedding
from src.layers.mlp import MLP


class DCNv2(nn.Module):
    """Combine a full-matrix cross network and a parallel deep network."""

    def __init__(
        self,
        field_vocab_sizes: Mapping[str, int],
        *,
        embedding_dim: int,
        num_cross_layers: int,
        hidden_dims: Sequence[int],
        dropout: float,
        embedding_pooling: str = "weighted_mean",
    ) -> None:
        super().__init__()
        missing_fields = set(ALL_FIELD_IDS) - set(field_vocab_sizes)
        extra_fields = set(field_vocab_sizes) - set(ALL_FIELD_IDS)
        if missing_fields or extra_fields:
            raise ValueError(
                "field vocabulary mismatch: "
                f"missing={sorted(missing_fields)}, extra={sorted(extra_fields)}"
            )
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.field_ids = ALL_FIELD_IDS
        self.embedding_dim = embedding_dim
        self.embedding_pooling = embedding_pooling
        self.num_cross_layers = num_cross_layers
        self.feature_embeddings = nn.ModuleDict(
            {
                field_id: SparseFeatureEmbedding(
                    int(field_vocab_sizes[field_id]),
                    embedding_dim,
                    pooling=embedding_pooling,
                )
                for field_id in self.field_ids
            }
        )
        input_dim = len(self.field_ids) * embedding_dim
        self.input_dim = input_dim
        self.cross_network = CrossNetworkV2(input_dim, num_cross_layers)
        self.mlp = MLP(input_dim, hidden_dims, dropout=dropout)
        self.output = nn.Linear(input_dim + self.mlp.output_dim, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @classmethod
    def from_feature_encoder(
        cls,
        encoder: FeatureEncoder,
        *,
        embedding_dim: int,
        num_cross_layers: int,
        hidden_dims: Sequence[int],
        dropout: float,
        embedding_pooling: str = "weighted_mean",
    ) -> "DCNv2":
        return cls(
            {
                field_id: encoder.field_vocabularies[field_id].vocab_size
                for field_id in ALL_FIELD_IDS
            },
            embedding_dim=embedding_dim,
            num_cross_layers=num_cross_layers,
            hidden_dims=hidden_dims,
            dropout=dropout,
            embedding_pooling=embedding_pooling,
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        features = batch.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("batch must contain a features mapping")

        dense_embeddings: list[Tensor] = []
        for field_id in self.field_ids:
            field = features.get(field_id)
            if not isinstance(field, Mapping):
                raise ValueError(f"batch is missing feature field {field_id}")
            try:
                ids = field["ids"]
                values = field["values"]
                offsets = field["offsets"]
            except KeyError as error:
                raise ValueError(
                    f"field {field_id} must contain ids, values and offsets"
                ) from error
            dense_embeddings.append(
                self.feature_embeddings[field_id](ids, values, offsets)
            )

        field_embeddings = torch.stack(dense_embeddings, dim=1)
        model_input = field_embeddings.flatten(start_dim=1)
        crossed = self.cross_network(model_input)
        deep = self.mlp(model_input)
        ctr_logit = self.output(torch.cat((crossed, deep), dim=1)).squeeze(-1)
        return {
            "ctr_logit": ctr_logit,
            "ctr": torch.sigmoid(ctr_logit),
        }
