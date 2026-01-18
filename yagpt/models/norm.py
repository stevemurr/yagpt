"""
RMSNorm - Root Mean Square Layer Normalization.

RMSNorm is a simplification of LayerNorm that removes the mean-centering step,
making it faster while maintaining similar performance. Used in LLaMA, Gemma, etc.

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm doesn't center the activations (no mean subtraction),
    which makes it ~10-15% faster while achieving similar results.

    Args:
        dim: The dimension to normalize over (typically embedding dimension)
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Using torch's built-in rms_norm for efficiency
        return torch.nn.functional.rms_norm(x, self.weight.shape, self.weight, self.eps)


def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Functional RMSNorm without learnable parameters.

    Useful for normalizing intermediate values (e.g., Q/K in attention).
    """
    return torch.nn.functional.rms_norm(x, (x.size(-1),), eps=eps)
