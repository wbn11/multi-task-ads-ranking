"""Factorization Machine interaction layer."""

from __future__ import annotations

from torch import Tensor, nn


class FactorizationMachine(nn.Module):
    """Compute summed pairwise field interactions in linear time."""

    def forward(self, field_embeddings: Tensor) -> Tensor:
        if field_embeddings.ndim != 3:
            raise ValueError(
                "field_embeddings must have shape [batch, fields, embedding_dim]"
            )
        if field_embeddings.shape[1] < 2:
            raise ValueError("FactorizationMachine requires at least two fields")

        summed_embeddings = field_embeddings.sum(dim=1)
        square_of_sum = summed_embeddings.square()
        sum_of_square = field_embeddings.square().sum(dim=1)
        return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
