"""Embedding layers for ragged categorical features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SparseLinearEmbedding(nn.Module):
    """Map one sparse field to a scalar linear contribution per sample.

    The input uses the flattened ``ids``/``values`` and ``offsets`` produced by
    :func:`src.data.dataset.collate_ads_batch`.  ``EmbeddingBag`` performs the
    segment reduction directly, so no padded two-dimensional tensor is built.
    """

    def __init__(self, vocab_size: int, *, padding_idx: int = 0) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if padding_idx < 0 or padding_idx >= vocab_size:
            raise ValueError("padding_idx must be inside the vocabulary")

        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=1,
            mode="sum",
            padding_idx=padding_idx,
            include_last_offset=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Zero initialization makes the first prediction exactly 0.5 and keeps
        # the LR baseline easy to interpret.
        nn.init.zeros_(self.embedding.weight)

    def forward(self, ids: Tensor, values: Tensor, offsets: Tensor) -> Tensor:
        if ids.ndim != 1 or values.ndim != 1 or offsets.ndim != 1:
            raise ValueError("ids, values and offsets must all be one-dimensional")
        if ids.shape != values.shape:
            raise ValueError("ids and values must have identical shapes")
        if offsets.numel() < 2:
            raise ValueError("offsets must contain at least one bag")
        if ids.dtype != torch.long or offsets.dtype != torch.long:
            raise TypeError("ids and offsets must use torch.long")
        if not values.is_floating_point():
            raise TypeError("values must use a floating-point dtype")

        return self.embedding(ids, offsets, per_sample_weights=values)


class SparseFeatureEmbedding(nn.Module):
    """Pool a ragged sparse field into one dense embedding per sample."""

    SUPPORTED_POOLING = ("sum", "weighted_mean")

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        pooling: str = "weighted_mean",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if pooling not in self.SUPPORTED_POOLING:
            raise ValueError(
                f"unsupported pooling={pooling!r}; supported={self.SUPPORTED_POOLING}"
            )
        if padding_idx < 0 or padding_idx >= vocab_size:
            raise ValueError("padding_idx must be inside the vocabulary")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.padding_idx = padding_idx
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            mode="sum",
            padding_idx=padding_idx,
            include_last_offset=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].zero_()

    def lookup_tokens(self, ids: Tensor) -> Tensor:
        """Return unpooled token embeddings using the EmbeddingBag weights."""

        if ids.ndim != 1:
            raise ValueError("ids must be one-dimensional")
        if ids.dtype != torch.long:
            raise TypeError("ids must use torch.long")
        return F.embedding(
            ids,
            self.embedding.weight,
            padding_idx=self.padding_idx,
        )

    def forward(self, ids: Tensor, values: Tensor, offsets: Tensor) -> Tensor:
        if ids.ndim != 1 or values.ndim != 1 or offsets.ndim != 1:
            raise ValueError("ids, values and offsets must all be one-dimensional")
        if ids.shape != values.shape:
            raise ValueError("ids and values must have identical shapes")
        if offsets.numel() < 2:
            raise ValueError("offsets must contain at least one bag")
        if ids.dtype != torch.long or offsets.dtype != torch.long:
            raise TypeError("ids and offsets must use torch.long")
        if not values.is_floating_point():
            raise TypeError("values must use a floating-point dtype")

        pooled = self.embedding(ids, offsets, per_sample_weights=values)
        if self.pooling == "sum":
            return pooled

        # Reduce each bag independently.  A global prefix sum followed by
        # subtraction is numerically unstable for large Arrow batches: after
        # millions of weighted history tokens, adding a small value can be
        # rounded away and a later one-token bag can receive a zero divisor.
        # Segment reduction keeps accumulation local to each sample without
        # materializing a padded [batch_size, max_history_length] tensor.
        bag_lengths = offsets[1:] - offsets[:-1]
        normalizer = torch.segment_reduce(
            values.abs(),
            reduce="sum",
            lengths=bag_lengths,
        ).clamp_min(torch.finfo(values.dtype).eps)
        return pooled / normalizer.unsqueeze(-1)
