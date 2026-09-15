"""Sparse logistic-regression baseline for CTR prediction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.data.feature_encoder import FeatureEncoder
from src.data.feature_schema import ALL_FIELD_IDS
from src.layers.embedding import SparseLinearEmbedding


class LogisticRegressionCTR(nn.Module):
    """Add independent sparse-field contributions and predict click probability."""

    def __init__(self, field_vocab_sizes: Mapping[str, int]) -> None:
        super().__init__()
        missing_fields = set(ALL_FIELD_IDS) - set(field_vocab_sizes)
        extra_fields = set(field_vocab_sizes) - set(ALL_FIELD_IDS)
        if missing_fields or extra_fields:
            raise ValueError(
                "field vocabulary mismatch: "
                f"missing={sorted(missing_fields)}, extra={sorted(extra_fields)}"
            )

        self.field_ids = ALL_FIELD_IDS
        self.field_weights = nn.ModuleDict(
            {
                field_id: SparseLinearEmbedding(int(field_vocab_sizes[field_id]))
                for field_id in self.field_ids
            }
        )
        self.bias = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_feature_encoder(cls, encoder: FeatureEncoder) -> "LogisticRegressionCTR":
        return cls(
            {
                field_id: encoder.field_vocabularies[field_id].vocab_size
                for field_id in ALL_FIELD_IDS
            }
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        features = batch.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("batch must contain a features mapping")

        field_scores: list[Tensor] = []
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
            field_scores.append(
                self.field_weights[field_id](ids, values, offsets)
            )

        ctr_logit = torch.stack(field_scores, dim=0).sum(dim=0).squeeze(-1)
        ctr_logit = ctr_logit + self.bias
        return {
            "ctr_logit": ctr_logit,
            "ctr": torch.sigmoid(ctr_logit),
        }
