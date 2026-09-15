"""Candidate-aware pooling for ragged weighted behavior histories."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TargetAwareHistoryPooling(nn.Module):
    """Activate one ragged history with its sample's target embedding.

    ``history_offsets`` partitions the flattened history tokens into one bag
    per target row.  The count prior makes the zero-initialized activation
    equivalent to a normalized weighted mean for positive feature values;
    training can then learn a target-dependent residual over that baseline.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        attention_hidden_dim: int,
        count_prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if attention_hidden_dim <= 0:
            raise ValueError("attention_hidden_dim must be positive")
        if count_prior_strength < 0.0:
            raise ValueError("count_prior_strength cannot be negative")

        self.embedding_dim = int(embedding_dim)
        self.attention_hidden_dim = int(attention_hidden_dim)
        self.count_prior_strength = float(count_prior_strength)
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.Sequential(
            nn.Linear(embedding_dim * 4, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.zeros_(self.key_projection.bias)
        first = self.activation[0]
        output = self.activation[2]
        assert isinstance(first, nn.Linear)
        assert isinstance(output, nn.Linear)
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        # Start from count-weighted pooling.  This makes the new layer a
        # learnable residual over the proven baseline rather than an unrelated
        # randomly initialized replacement.
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def _validate_inputs(
        self,
        query: Tensor,
        history_embeddings: Tensor,
        history_values: Tensor,
        history_offsets: Tensor,
    ) -> Tensor:
        if query.ndim != 2 or query.shape[1] != self.embedding_dim:
            raise ValueError(
                f"query must have shape [batch, {self.embedding_dim}]"
            )
        if (
            history_embeddings.ndim != 2
            or history_embeddings.shape[1] != self.embedding_dim
        ):
            raise ValueError(
                "history_embeddings must have shape "
                f"[tokens, {self.embedding_dim}]"
            )
        if history_values.ndim != 1 or history_offsets.ndim != 1:
            raise ValueError("history_values and history_offsets must be 1-D")
        if history_embeddings.shape[0] != history_values.numel():
            raise ValueError("history embedding/value counts must match")
        if history_offsets.dtype != torch.long:
            raise TypeError("history_offsets must use torch.long")
        if history_offsets.numel() != query.shape[0] + 1:
            raise ValueError("history_offsets length must equal batch_size + 1")
        lengths = history_offsets[1:] - history_offsets[:-1]
        # Content checks are useful for CPU-side unit tests but would introduce
        # four host/device synchronizations per GPU batch.  The collator owns
        # these invariants in production, so CUDA forwards keep this path fully
        # asynchronous.
        if history_offsets.device.type == "cpu":
            if int(history_offsets[0].item()) != 0:
                raise ValueError("history_offsets must start at zero")
            if int(history_offsets[-1].item()) != history_values.numel():
                raise ValueError("history_offsets must end at the token count")
            if bool((lengths <= 0).any()):
                raise ValueError(
                    "every history bag must contain a token; the DataLoader "
                    "should replace missing histories with field-local UNK"
                )
        return lengths

    def forward(
        self,
        query: Tensor,
        history_embeddings: Tensor,
        history_values: Tensor,
        history_offsets: Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        lengths = self._validate_inputs(
            query,
            history_embeddings,
            history_values,
            history_offsets,
        )
        projected_queries = self.query_projection(query)
        token_queries = torch.repeat_interleave(
            projected_queries,
            lengths,
            dim=0,
        )
        token_keys = self.key_projection(history_embeddings)
        interactions = torch.cat(
            (
                token_queries,
                token_keys,
                token_queries * token_keys,
                token_queries - token_keys,
            ),
            dim=-1,
        )

        # Compute normalization in float32 even under AMP.  Long histories can
        # otherwise overflow exp or lose small denominators in float16.
        scores = self.activation(interactions).squeeze(-1).float()
        if self.count_prior_strength:
            count_prior = torch.log(
                history_values.float().abs().clamp_min(1.0e-12)
            )
            scores = scores + self.count_prior_strength * count_prior

        segment_max = torch.segment_reduce(
            scores,
            reduce="max",
            lengths=lengths,
        )
        stabilized = scores - torch.repeat_interleave(segment_max, lengths)
        unnormalized = torch.exp(stabilized)
        segment_sum = torch.segment_reduce(
            unnormalized,
            reduce="sum",
            lengths=lengths,
        ).clamp_min(torch.finfo(unnormalized.dtype).tiny)
        attention_weights = unnormalized / torch.repeat_interleave(
            segment_sum,
            lengths,
        )
        pooled = torch.segment_reduce(
            history_embeddings.float() * attention_weights.unsqueeze(-1),
            reduce="sum",
            lengths=lengths,
        ).to(history_embeddings.dtype)

        if return_attention_weights:
            return pooled, attention_weights
        return pooled
