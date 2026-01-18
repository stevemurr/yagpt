"""
Feed-Forward Network with SwiGLU activation.

SwiGLU is a gated linear unit variant that uses the Swish (SiLU) activation
function. It has become the standard for modern LLMs (LLaMA, PaLM, etc.)
as it provides better performance than standard ReLU or GELU MLPs.

Reference: https://arxiv.org/abs/2002.05202
"""

import torch.nn as nn
from torch import Tensor


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Architecture:
        x -> [W_gate, W_up] -> SiLU(gate) * up -> W_down -> out

    The gating mechanism allows the network to selectively pass information,
    which empirically improves model quality.

    Args:
        dim: Input and output dimension (embedding size)
        hidden_dim: Hidden dimension (typically 4 * dim, but can be customized)
        bias: Whether to use bias in linear layers (modern LLMs typically don't)
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim

        # Combined gate and up projection for efficiency
        # Output is 2 * hidden_dim, split into gate and up
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # Project and split into gate and up paths
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # SwiGLU: SiLU(gate) * up, then project down
        return self.down(nn.functional.silu(gate) * up)
