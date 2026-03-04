"""
QLoRA - 4-bit NormalFloat Quantization.

Implements NF4 quantization from scratch (no bitsandbytes dependency),
making it compatible with aarch64 / DGX Spark.

NF4 maps 16 quantile-spaced values to the normal distribution, providing
better representation of normally-distributed weights than uniform quantization.

Reference: https://arxiv.org/abs/2305.14314
"""

import torch
import torch.nn as nn
from torch import Tensor

# NF4 quantile levels: optimal 4-bit codes for normally-distributed values.
# These are the 16 quantiles of N(0,1) that minimize quantization error.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0,
], dtype=torch.float32)

# Block size for per-block quantization
DEFAULT_BLOCK_SIZE = 64


class NF4Linear(nn.Module):
    """
    4-bit NormalFloat quantized linear layer.

    Stores weights in NF4 format with per-block absmax scaling.
    The layer is frozen (no gradients on quantized weights).

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias
        block_size: Number of elements per quantization block
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Number of blocks
        total_elements = in_features * out_features
        n_blocks = (total_elements + block_size - 1) // block_size

        # Quantized weights stored as uint8 (two 4-bit values per byte)
        n_bytes = (total_elements + 1) // 2
        self.register_buffer("quantized_weight", torch.zeros(n_bytes, dtype=torch.uint8))

        # Per-block absmax scaling factors
        self.register_buffer("scales", torch.ones(n_blocks, dtype=torch.float16))

        # Original shape for reconstruction
        self.register_buffer("_shape", torch.tensor([out_features, in_features]))

        # Optional bias (kept in full precision)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # NF4 lookup table
        self.register_buffer("nf4_levels", NF4_LEVELS)

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = DEFAULT_BLOCK_SIZE) -> "NF4Linear":
        """Create an NF4 layer from an existing nn.Linear."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
        )

        # Quantize the weight
        weight = linear.weight.data.float().flatten()
        _quantize_to_nf4(weight, layer.quantized_weight, layer.scales, block_size, layer.nf4_levels)

        # Copy bias if present
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.data.clone())

        return layer

    def dequantize(self) -> Tensor:
        """Dequantize weight back to float."""
        out_features, in_features = self._shape[0].item(), self._shape[1].item()
        total = out_features * in_features

        return _dequantize_nf4(
            self.quantized_weight, self.scales, total,
            self.block_size, self.nf4_levels,
        ).view(out_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        # Dequantize weight on-the-fly
        weight = self.dequantize().to(x.dtype)
        return nn.functional.linear(x, weight, self.bias)


def _quantize_to_nf4(
    flat_weight: Tensor,
    out_quantized: Tensor,
    out_scales: Tensor,
    block_size: int,
    nf4_levels: Tensor,
) -> None:
    """Quantize a flat weight tensor to NF4 in-place."""
    total = flat_weight.numel()
    nf4 = nf4_levels.to(flat_weight.device)

    codes = torch.zeros(total, dtype=torch.uint8, device=flat_weight.device)

    for block_idx in range(0, total, block_size):
        block_end = min(block_idx + block_size, total)
        block = flat_weight[block_idx:block_end]

        # Per-block absmax scaling
        absmax = block.abs().max().clamp(min=1e-12)
        out_scales[block_idx // block_size] = absmax.half()

        # Normalize to [-1, 1]
        normalized = block / absmax

        # Find nearest NF4 level for each value
        distances = (normalized.unsqueeze(-1) - nf4.unsqueeze(0)).abs()
        block_codes = distances.argmin(dim=-1).to(torch.uint8)
        codes[block_idx:block_end] = block_codes

    # Pack two 4-bit codes per byte
    n_pairs = total // 2
    out_quantized[:n_pairs] = (codes[0::2][:n_pairs] << 4) | codes[1::2][:n_pairs]
    if total % 2 == 1:
        out_quantized[n_pairs] = codes[-1] << 4


def _dequantize_nf4(
    quantized: Tensor,
    scales: Tensor,
    total: int,
    block_size: int,
    nf4_levels: Tensor,
) -> Tensor:
    """Dequantize NF4 packed data back to float."""
    device = quantized.device
    nf4 = nf4_levels.to(device)

    # Unpack 4-bit codes
    codes = torch.zeros(total, dtype=torch.uint8, device=device)
    n_pairs = total // 2
    codes[0::2][:n_pairs] = (quantized[:n_pairs] >> 4) & 0x0F
    codes[1::2][:n_pairs] = quantized[:n_pairs] & 0x0F
    if total % 2 == 1:
        codes[-1] = (quantized[n_pairs] >> 4) & 0x0F

    # Look up NF4 values
    values = nf4[codes.long()]

    # Apply per-block scaling
    result = torch.zeros(total, dtype=torch.float32, device=device)
    for block_idx in range(0, total, block_size):
        block_end = min(block_idx + block_size, total)
        scale = scales[block_idx // block_size].float()
        result[block_idx:block_end] = values[block_idx:block_end] * scale

    return result


def quantize_model_nf4(
    model: nn.Module,
    block_size: int = DEFAULT_BLOCK_SIZE,
    skip_modules: set[str] | None = None,
) -> nn.Module:
    """
    Quantize all Linear layers in a model to NF4.

    Replaces nn.Linear with NF4Linear (frozen). Typically used before
    applying LoRA so that the base model is quantized but LoRA adapters
    remain in full precision.

    Args:
        model: Model to quantize (modified in-place)
        block_size: Quantization block size
        skip_modules: Module name suffixes to skip (e.g., {"lm_head"})

    Returns:
        The modified model
    """
    if skip_modules is None:
        skip_modules = {"lm_head"}  # Don't quantize the output head

    for name, module in list(model.named_modules()):
        short_name = name.split(".")[-1] if "." in name else name
        if short_name in skip_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Replace with NF4
        nf4_module = NF4Linear.from_linear(module, block_size=block_size)

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nf4_module)

    return model
