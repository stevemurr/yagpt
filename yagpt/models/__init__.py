"""
YAGPT Models - Clean, modular transformer components.

This module provides the building blocks for GPT-style language models:
- GPT: Main model class
- GPTConfig: Model configuration
- Individual components (attention, mlp, norm, rope) for customization
"""

from .attention import CausalAttention
from .gpt import GPT, GPTConfig
from .mlp import SwiGLU
from .norm import RMSNorm, rms_norm
from .rope import apply_rope, build_rope_cache

__all__ = [
    # Main model
    "GPT",
    "GPTConfig",
    # Components
    "CausalAttention",
    "SwiGLU",
    "RMSNorm",
    "rms_norm",
    "apply_rope",
    "build_rope_cache",
]
