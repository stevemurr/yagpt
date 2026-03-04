"""
LoRA - Low-Rank Adaptation.

Adds trainable low-rank matrices A and B to frozen linear layers:
    output = W @ x + (B @ A) @ x * (alpha / rank)

B is initialized to zeros so the LoRA starts as identity (no change).

Reference: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Wraps an existing nn.Linear or NF4Linear with low-rank A/B matrices.
    The base weight is frozen; only A and B are trainable.

    Args:
        base: The original nn.Linear or NF4Linear to wrap
        rank: LoRA rank (typically 4-64)
        alpha: LoRA scaling factor (typically 2*rank)
        dropout: Dropout on LoRA path (0.0 = none)
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base.in_features
        out_features = base.out_features

        # Freeze base weights
        for p in base.parameters():
            p.requires_grad_(False)

        # LoRA matrices: A projects down, B projects up
        # A: (rank, in_features) - initialized with Kaiming
        # B: (out_features, rank) - initialized to zeros (identity at start)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Base frozen path
        base_out = self.base(x)

        # LoRA path: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.2f}"


# Default modules to apply LoRA to in a GPT model
DEFAULT_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "out_proj", "gate_up", "down"}


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    target_modules: set[str] | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to matching Linear layers in a model.

    Walks the module tree and replaces Linear layers whose name matches
    one of the target module names.

    Args:
        model: Model to modify (modified in-place)
        rank: LoRA rank
        alpha: LoRA alpha (default: 2 * rank)
        target_modules: Set of module name suffixes to target
        dropout: Dropout on LoRA path

    Returns:
        The modified model (same object, modified in-place)
    """
    if alpha is None:
        alpha = 2.0 * rank
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    from yagpt.lora.qlora import NF4Linear

    for name, module in list(model.named_modules()):
        # Check if this module's name ends with a target name
        short_name = name.split(".")[-1] if "." in name else name
        if short_name not in target_modules:
            continue
        if not isinstance(module, (nn.Linear, NF4Linear)):
            continue

        # Replace with LoRA wrapper
        lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)

        # Set the LoRA module in the parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_module)

    return model


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Extract only LoRA parameters from a model's state dict."""
    return {
        name: param
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def count_lora_params(model: nn.Module) -> tuple[int, int]:
    """
    Count LoRA vs total parameters.

    Returns:
        Tuple of (lora_params, total_params)
    """
    lora_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora_A" in name or "lora_B" in name:
            lora_params += param.numel()

    return lora_params, total_params
