"""DeepFM CTR model for ragged Ali-CCP categorical features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from src.data.feature_encoder import FeatureEncoder
from src.data.feature_schema import ALL_FIELD_IDS
from src.layers.embedding import SparseFeatureEmbedding, SparseLinearEmbedding
from src.layers.fm import FactorizationMachine
from src.layers.mlp import MLP


class DeepFM(nn.Module):
    """Combine first-order, FM second-order and deep nonlinear contributions."""

    def __init__(
        self,
        field_vocab_sizes: Mapping[str, int],
        *,
        embedding_dim: int,
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
        self.linear_weights = nn.ModuleDict(
            {
                field_id: SparseLinearEmbedding(int(field_vocab_sizes[field_id]))
                for field_id in self.field_ids
            }
        )
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
        self.fm = FactorizationMachine()
        self.mlp = MLP(
            len(self.field_ids) * embedding_dim,
            hidden_dims,
            dropout=dropout,
        )
        self.deep_output = nn.Linear(self.mlp.output_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.deep_output.weight)

    @classmethod
    def from_feature_encoder(
        cls,
        encoder: FeatureEncoder,
        *,
        embedding_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        embedding_pooling: str = "weighted_mean",
    ) -> "DeepFM":
        return cls(
            {
                field_id: encoder.field_vocabularies[field_id].vocab_size
                for field_id in ALL_FIELD_IDS
            },
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            embedding_pooling=embedding_pooling,
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        features = batch.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("batch must contain a features mapping")

        linear_scores: list[Tensor] = []
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
            linear_scores.append(
                self.linear_weights[field_id](ids, values, offsets)
            )
            dense_embeddings.append(
                self.feature_embeddings[field_id](ids, values, offsets)
            )

        linear_logit = torch.stack(linear_scores, dim=0).sum(dim=0).squeeze(-1)
        field_embeddings = torch.stack(dense_embeddings, dim=1)
        fm_logit = self.fm(field_embeddings)
        deep_input = field_embeddings.flatten(start_dim=1)
        deep_logit = self.deep_output(self.mlp(deep_input)).squeeze(-1)
        ctr_logit = linear_logit + fm_logit + deep_logit + self.bias
        return {
            "ctr_logit": ctr_logit,
            "ctr": torch.sigmoid(ctr_logit),
        }
