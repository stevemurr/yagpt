"""
GPT Model - A clean, configurable GPT implementation.

This is the main model class that composes all the components into a complete
GPT-style language model. Supports modern features like GQA, RoPE, RMSNorm,
and SwiGLU while keeping the code readable.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .attention import CausalAttention
from .mlp import SwiGLU
from .norm import RMSNorm
from .rope import build_rope_cache


@dataclass
class GPTConfig:
    """Configuration for GPT model."""

    # Model architecture
    vocab_size: int = 50257  # GPT-2 vocab size, override for other tokenizers
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None  # For GQA. None = same as n_heads (MHA)
    dim: int = 768
    hidden_dim: int | None = None  # MLP hidden dim. None = 4 * dim
    max_seq_len: int = 2048

    # Regularization
    dropout: float = 0.0  # Dropout rate (0.0 for pretraining)

    # Initialization
    init_std: float = 0.02  # Std for weight initialization

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.dim


class Block(nn.Module):
    """
    Transformer block with pre-normalization.

    Architecture:
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
        |___________________________|   |__________________|
                residual                     residual
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = CausalAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
        )
        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = SwiGLU(dim=config.dim, hidden_dim=config.hidden_dim)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        return_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        # Attention with residual
        h, new_kv_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache, return_cache)
        x = x + h

        # MLP with residual
        x = x + self.mlp(self.mlp_norm(x))

        return x, new_kv_cache


class GPT(nn.Module):
    """
    GPT Language Model.

    A decoder-only transformer for causal language modeling.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no position embeddings - we use RoPE)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        # Output
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.tok_emb.weight = self.lm_head.weight

        # Precompute RoPE cache
        head_dim = config.dim // config.n_heads
        self._rope_cache: tuple[Tensor, Tensor] | None = None
        self._cache_seq_len = 0

        # Initialize weights
        self.apply(self._init_weights)
        self._init_residual_scaling()

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small std."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _init_residual_scaling(self):
        """Scale residual projections by 1/sqrt(2*n_layers) for stability."""
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=self.config.init_std * scale)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=self.config.init_std * scale)

    def _get_rope_cache(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Get or build RoPE cache for the given sequence length."""
        if self._rope_cache is None or seq_len > self._cache_seq_len:
            head_dim = self.config.dim // self.config.n_heads
            # Build cache for longer sequences to avoid rebuilding
            cache_len = max(seq_len, self.config.max_seq_len)
            cos, sin = build_rope_cache(cache_len, head_dim, device=device)
            self._rope_cache = (cos, sin)
            self._cache_seq_len = cache_len

        cos, sin = self._rope_cache
        return cos[:seq_len], sin[:seq_len]

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        kv_cache: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, Tensor | None, list[tuple[Tensor, Tensor]] | None]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            targets: Target token IDs for loss computation (batch, seq_len)
            kv_cache: List of (k, v) tuples per layer for generation

        Returns:
            Tuple of (logits, loss, new_kv_cache)
            - logits: (batch, seq_len, vocab_size) or (batch, 1, vocab_size) with cache
            - loss: Scalar loss if targets provided, else None
            - new_kv_cache: Updated KV cache if input cache was provided
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Get RoPE embeddings
        # For KV cache, we need positions starting from cache length
        start_pos = 0 if kv_cache is None else kv_cache[0][0].shape[1]
        cos, sin = self._get_rope_cache(start_pos + seq_len, device)
        cos, sin = cos[start_pos:start_pos + seq_len], sin[start_pos:start_pos + seq_len]

        # Token embeddings
        x = self.tok_emb(input_ids)

        # Determine if we should return KV cache (for generation, not training)
        return_cache = targets is None

        # Transformer blocks
        new_kv_cache = [] if return_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_layer_cache = block(x, cos, sin, layer_cache, return_cache)
            if new_kv_cache is not None:
                new_kv_cache.append(new_layer_cache)

        # Output projection
        x = self.norm(x)

        # Compute logits
        if targets is not None:
            # Training: compute full logits for loss
            logits = self.lm_head(x)
            logits = logits.float()  # Upcast for stable loss computation
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute last token logits
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_kv_cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        # Prefill: process all input tokens at once to build KV cache
        logits, _, kv_cache = self(input_ids)

        for _ in range(max_new_tokens):
            # Get next token logits
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Forward with KV cache (only process new token)
            logits, _, kv_cache = self(next_token, kv_cache=kv_cache)

        return input_ids

    def num_parameters(self, exclude_embeddings: bool = True) -> int:
        """Count model parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            # Embeddings are weight-tied with lm_head, only count once
            n_params -= self.tok_emb.weight.numel()
        return n_params
