"""
Rotary Position Embeddings (RoPE).

RoPE encodes position information by rotating pairs of dimensions in the
query and key vectors. This allows the model to generalize to longer sequences
and has become the standard for modern LLMs (LLaMA, GPT-NeoX, etc.).

Reference: https://arxiv.org/abs/2104.09864
"""

import torch
from torch import Tensor


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: int = 10000,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Precompute cosine and sine values for rotary embeddings.

    Args:
        seq_len: Maximum sequence length to cache
        head_dim: Dimension of each attention head (must be even)
        base: Base for the frequency computation (10000 is standard)
        dtype: Data type for the cache (use float32 for precision)
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors, each of shape (seq_len, head_dim)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Compute inverse frequencies: 1 / (base^(2i/d)) for i in [0, d/2)
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))

    # Compute position * frequency for all positions
    positions = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, half_dim)

    # Duplicate frequencies for pairing: [f0, f1, ...] -> [f0, f0, f1, f1, ...]
    freqs = freqs.repeat_interleave(2, dim=-1)  # (seq_len, head_dim)

    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim)
        cos: Cosine cache of shape (seq_len, head_dim) or broadcastable
        sin: Sine cache of shape (seq_len, head_dim) or broadcastable

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    # Reshape for rotation: split into pairs
    # x: (batch, seq, heads, dim) -> pairs of adjacent elements
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices

    # Reshape cos/sin to broadcast: (seq, dim) -> (1, seq, 1, dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Only use the frequencies we need
    cos = cos[..., 0::2]
    sin = sin[..., 0::2]

    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Interleave back: [r1_0, r2_0, r1_1, r2_1, ...]
    out = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

    return out.type_as(x)
