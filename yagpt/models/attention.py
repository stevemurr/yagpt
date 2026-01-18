"""
Causal Self-Attention with Grouped Query Attention (GQA) support.

This module implements efficient multi-head attention with:
- Grouped Query Attention (GQA): Shares K/V heads across Q heads for efficiency
- Flash Attention: Uses PyTorch's optimized SDPA implementation
- KV Caching: Efficient autoregressive generation

References:
- Attention: https://arxiv.org/abs/1706.03762
- GQA: https://arxiv.org/abs/2305.13245
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope import apply_rope


class CausalAttention(nn.Module):
    """
    Causal Self-Attention with optional Grouped Query Attention.

    Args:
        dim: Model dimension (embedding size)
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (for GQA). If None, equals n_heads (MHA)
        head_dim: Dimension per head. If None, computed as dim // n_heads
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or dim // n_heads

        # Validate configuration
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        assert n_heads % self.n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )

        # Projections
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        return_cache: bool = True,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """
        Forward pass with optional KV caching for generation.

        Args:
            x: Input tensor (batch, seq_len, dim)
            cos: Cosine values for RoPE (seq_len, head_dim)
            sin: Sine values for RoPE (seq_len, head_dim)
            kv_cache: Optional tuple of (cached_k, cached_v) for generation
            return_cache: Whether to return KV cache (for generation)

        Returns:
            Tuple of (output, new_kv_cache)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        # Store cache before transpose (in original shape for concatenation)
        new_kv_cache = (k, v) if return_cache else None

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with Flash Attention
        # enable_gqa=True handles the head broadcasting for GQA
        is_causal = kv_cache is None  # Only causal during training/prefill
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            enable_gqa=(self.n_heads != self.n_kv_heads),
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(y), new_kv_cache
