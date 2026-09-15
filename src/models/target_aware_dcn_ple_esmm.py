"""DCN-PLE-ESMM with candidate-aware pooling for behavior histories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from src.data.feature_encoder import FeatureEncoder
from src.layers.target_aware_pooling import TargetAwareHistoryPooling

from .dcn_ple_esmm import DCNPLEESMM


# The field semantics come from the Ali-CCP schema rather than guesses about
# anonymous fields.  Each history is activated by the corresponding current
# impression-side target field.  Combination fields remain in the model for a
# controlled comparison with the original final candidate.
HISTORY_TARGET_FIELD_PAIRS: tuple[tuple[str, str], ...] = (
    ("109_14", "206"),
    ("110_14", "207"),
    ("127_14", "216"),
    ("150_14", "210"),
)


class TargetAwareDCNPLEESMM(DCNPLEESMM):
    """Replace four history means with target-conditioned ragged attention.

    Every field still contributes exactly one ``embedding_dim`` vector, so the
    downstream DCNv2, PLE experts, task gates, towers and ESMM objective are
    unchanged.  This isolates the experiment to history representation.
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
        history_attention_hidden_dim: int,
        history_count_prior_strength: float = 1.0,
        gate_dropout: float = 0.0,
        embedding_pooling: str = "weighted_mean",
    ) -> None:
        super().__init__(
            field_vocab_sizes,
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
        self.history_attention_hidden_dim = int(history_attention_hidden_dim)
        self.history_count_prior_strength = float(history_count_prior_strength)
        self.history_target_field_pairs = HISTORY_TARGET_FIELD_PAIRS
        missing_fields = {
            field_id
            for pair in HISTORY_TARGET_FIELD_PAIRS
            for field_id in pair
            if field_id not in self.field_ids
        }
        if missing_fields:
            raise ValueError(
                "history attention fields are absent from the schema: "
                f"{sorted(missing_fields)}"
            )
        self.history_attention = nn.ModuleDict(
            {
                history_field: TargetAwareHistoryPooling(
                    embedding_dim,
                    attention_hidden_dim=history_attention_hidden_dim,
                    count_prior_strength=history_count_prior_strength,
                )
                for history_field, _ in HISTORY_TARGET_FIELD_PAIRS
            }
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
        history_attention_hidden_dim: int,
        history_count_prior_strength: float = 1.0,
        gate_dropout: float = 0.0,
        embedding_pooling: str = "weighted_mean",
    ) -> "TargetAwareDCNPLEESMM":
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
            history_attention_hidden_dim=history_attention_hidden_dim,
            history_count_prior_strength=history_count_prior_strength,
            gate_dropout=gate_dropout,
            embedding_pooling=embedding_pooling,
        )

    @staticmethod
    def _field_tensors(
        features: Mapping[str, Any],
        field_id: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        if not all(isinstance(tensor, Tensor) for tensor in (ids, values, offsets)):
            raise TypeError(f"field {field_id} tensors must be torch.Tensor")
        return ids, values, offsets

    def _model_input(self, batch: Mapping[str, Any]) -> Tensor:
        features = batch.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("batch must contain a features mapping")

        history_fields = {
            history_field for history_field, _ in self.history_target_field_pairs
        }
        dense_embeddings: dict[str, Tensor] = {}

        # Pool ordinary fields, including the four current target fields, with
        # the existing configured reducer.
        for field_id in self.field_ids:
            if field_id in history_fields:
                continue
            ids, values, offsets = self._field_tensors(features, field_id)
            dense_embeddings[field_id] = self.feature_embeddings[field_id](
                ids,
                values,
                offsets,
            )

        # Keep each history ragged.  No [batch, max_history_length] padding is
        # materialized; offsets define the segment belonging to each sample.
        for history_field, target_field in self.history_target_field_pairs:
            ids, values, offsets = self._field_tensors(features, history_field)
            history_tokens = self.feature_embeddings[
                history_field
            ].lookup_tokens(ids)
            dense_embeddings[history_field] = self.history_attention[
                history_field
            ](
                dense_embeddings[target_field],
                history_tokens,
                values,
                offsets,
            )

        field_embeddings = torch.stack(
            [dense_embeddings[field_id] for field_id in self.field_ids],
            dim=1,
        )
        raw_input = field_embeddings.flatten(start_dim=1)
        crossed = self.cross_network(raw_input)
        return self.cross_normalization(crossed)
