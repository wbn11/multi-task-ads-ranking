"""Shared Bottom model for joint CTR and traditional clicked-space CVR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from src.data.feature_encoder import FeatureEncoder
from src.data.feature_schema import ALL_FIELD_IDS
from src.layers.embedding import SparseFeatureEmbedding
from src.layers.mlp import MLP


class SharedBottom(nn.Module):
    """Share embeddings and a bottom MLP before task-specific CTR/CVR towers."""

    def __init__(
        self,
        field_vocab_sizes: Mapping[str, int],
        *,
        embedding_dim: int,
        shared_hidden_dims: Sequence[int],
        tower_hidden_dims: Sequence[int],
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
        self.input_dim = len(self.field_ids) * embedding_dim
        self.shared_bottom = MLP(
            self.input_dim,
            shared_hidden_dims,
            dropout=dropout,
        )
        self.ctr_tower = MLP(
            self.shared_bottom.output_dim,
            tower_hidden_dims,
            dropout=dropout,
        )
        self.cvr_tower = MLP(
            self.shared_bottom.output_dim,
            tower_hidden_dims,
            dropout=dropout,
        )
        self.ctr_output = nn.Linear(self.ctr_tower.output_dim, 1)
        self.cvr_output = nn.Linear(self.cvr_tower.output_dim, 1)
        nn.init.xavier_uniform_(self.ctr_output.weight)
        nn.init.zeros_(self.ctr_output.bias)
        nn.init.xavier_uniform_(self.cvr_output.weight)
        nn.init.zeros_(self.cvr_output.bias)

    @classmethod
    def from_feature_encoder(
        cls,
        encoder: FeatureEncoder,
        *,
        embedding_dim: int,
        shared_hidden_dims: Sequence[int],
        tower_hidden_dims: Sequence[int],
        dropout: float,
        embedding_pooling: str = "weighted_mean",
    ) -> "SharedBottom":
        return cls(
            {
                field_id: encoder.field_vocabularies[field_id].vocab_size
                for field_id in ALL_FIELD_IDS
            },
            embedding_dim=embedding_dim,
            shared_hidden_dims=shared_hidden_dims,
            tower_hidden_dims=tower_hidden_dims,
            dropout=dropout,
            embedding_pooling=embedding_pooling,
        )

    def _model_input(self, batch: Mapping[str, Any]) -> Tensor:
        """Embed and flatten every schema field for downstream backbones."""

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
        return field_embeddings.flatten(start_dim=1)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        model_input = self._model_input(batch)
        shared_representation = self.shared_bottom(model_input)
        ctr_logit = self.ctr_output(
            self.ctr_tower(shared_representation)
        ).squeeze(-1)
        cvr_logit = self.cvr_output(
            self.cvr_tower(shared_representation)
        ).squeeze(-1)
        ctr = torch.sigmoid(ctr_logit)
        cvr = torch.sigmoid(cvr_logit)
        return {
            "ctr_logit": ctr_logit,
            "cvr_logit": cvr_logit,
            "ctr": ctr,
            "cvr": cvr,
            "ctcvr": ctr * cvr,
        }
