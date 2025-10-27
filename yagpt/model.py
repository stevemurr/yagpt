"""
GPT-4 Mini Model Implementation in PyTorch

A scaled-down GPT-4 architecture for educational purposes and experimentation.
Implements all the key components: multi-head attention, feedforward networks,
layer normalization, and positional embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """
    Configuration for GPT model.

    Default values are for a small GPT-4 mini suitable for experimentation.
    For reference, GPT-3 175B has: n_layer=96, n_head=96, n_embd=12288
    """
    # Model architecture
    vocab_size: int = 100277  # GPT-4 tokenizer vocabulary size (cl100k_base)
    n_layer: int = 12         # Number of transformer blocks
    n_head: int = 12          # Number of attention heads
    n_embd: int = 768         # Embedding dimension

    # Context and sequence length
    block_size: int = 2048    # Maximum sequence length (context window)

    # Regularization
    dropout: float = 0.1      # Dropout probability
    bias: bool = True         # Use bias in linear layers and LayerNorms

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    use_amp: bool = True      # Use Automatic Mixed Precision (2-3x speedup on modern GPUs)
    use_compile: bool = True  # Use torch.compile for fused operations (10-30% speedup, PyTorch 2.0+)

    # Positional encoding
    use_rope: bool = True     # Use Rotary Position Embeddings (better length generalization)

    def __post_init__(self):
        # Validate that n_embd is divisible by n_head
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE encodes position information by rotating the query and key vectors in the
    complex plane. This provides better length generalization compared to learned
    positional embeddings, allowing the model to handle sequences longer than
    those seen during training.

    Paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the frequency tensor (cached for efficiency)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin values for max sequence length
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute and cache cos/sin values for efficiency."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but uses a different permutation to get same results
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        """
        Apply rotary position embeddings to input tensor.

        Args:
            x: Input tensor of shape (batch, n_head, seq_len, head_dim)
            seq_len: Sequence length

        Returns:
            Tensor with rotary embeddings applied
        """
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            x * self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0) +
            self._rotate_half(x) * self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism with Flash Attention.

    This is the core component that allows the model to attend to previous tokens
    in the sequence. Uses "causal" masking to prevent attending to future tokens.

    Uses PyTorch's scaled_dot_product_attention which automatically selects the
    best implementation (Flash Attention 2, Memory-Efficient Attention, or math)
    based on hardware and inputs.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # Rotary Position Embeddings (if enabled)
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            self.rotary_emb = RotaryPositionalEmbedding(
                dim=head_dim,
                max_seq_len=config.block_size
            )

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Forward pass with optional KV caching for efficient autoregressive generation.

        Args:
            x: Input tensor of shape (B, T, C)
            past_kv: Optional tuple of (past_k, past_v) from previous forward pass
            use_cache: If True, return the current k,v for caching

        Returns:
            y: Output tensor of shape (B, T, C)
            present_kv: Optional tuple of (k, v) if use_cache=True, else None
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch
        # and move head dimension forward to be the batch dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # If using cache, concatenate with past key-values
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # Concatenate along sequence dimension
            v = torch.cat([past_v, v], dim=2)

        # Apply Rotary Position Embeddings to q and k (if enabled)
        # Note: For cached k, we need the full sequence length for position encoding
        if self.use_rope:
            kv_seq_len = k.size(2)  # Full sequence length including cache
            q_seq_len = q.size(2)   # Query sequence length (usually 1 during generation)

            # Apply RoPE with correct position offsets
            if past_kv is not None:
                # For cached inference, q is at position kv_seq_len - 1
                q = self.rotary_emb(q, kv_seq_len)[:, :, -q_seq_len:, :]
                # k already has RoPE from previous calls, only apply to new token
                k_new = self.rotary_emb(k[:, :, -1:, :], kv_seq_len)
                k = torch.cat([k[:, :, :-1, :], k_new], dim=2)
            else:
                # Normal forward pass
                q = self.rotary_emb(q, kv_seq_len)
                k = self.rotary_emb(k, kv_seq_len)

        # Flash Attention using PyTorch's scaled_dot_product_attention
        # Automatically selects the best kernel (Flash Attention 2, memory-efficient, or math)
        kv_seq_len = k.size(2)
        q_seq_len = q.size(2)

        # Only use is_causal when not using cache (cache handles causality)
        is_causal = (past_kv is None) and (kv_seq_len == q_seq_len)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=is_causal
        )

        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, q_seq_len, C)

        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))

        # Return k,v cache if requested
        present_kv = (k, v) if use_cache else None
        return y, present_kv


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network).

    A simple 2-layer feedforward network with GELU activation.
    GPT-4 uses a 4x expansion in the hidden dimension.

    Can be compiled with torch.compile for fused operations (Linear+GELU fusion).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer Block.

    The fundamental building block of GPT. Consists of:
    1. Layer normalization
    2. Multi-head causal self-attention
    3. Residual connection
    4. Layer normalization
    5. Feed-forward network (MLP)
    6. Residual connection

    Uses pre-normalization (LayerNorm before attention/MLP) which is more stable.
    Can be compiled with torch.compile for better fusion of operations.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Forward pass with optional KV caching.

        Args:
            x: Input tensor
            past_kv: Past key-value cache from attention
            use_cache: Whether to return present key-values

        Returns:
            x: Output tensor
            present_kv: Present key-values if use_cache=True, else None
        """
        # Pre-normalization: normalize before attention
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out

        # Pre-normalization: normalize before MLP
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class GPT(nn.Module):
    """
    GPT-4 Mini Language Model with Modern Optimizations.

    The full GPT model consisting of:
    - Token embeddings
    - Positional embeddings (Rotary or learned)
    - Transformer blocks with Flash Attention
    - Layer normalization
    - Language modeling head

    Optimizations:
    - Flash Attention: Faster attention computation via scaled_dot_product_attention
    - KV Cache: O(n) generation instead of O(n²) by caching key-value pairs
    - RoPE: Rotary Position Embeddings for better length generalization
    - Mixed Precision (AMP): 2-3x speedup on modern GPUs
    - torch.compile: Fused operations for MLP (10-30% speedup)

    Example usage with AMP:
        model = GPT(config)
        scaler = model.get_grad_scaler('cuda')

        # Training loop
        with torch.amp.autocast(device_type='cuda', enabled=config.use_amp):
            logits, loss, _ = model(x, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    Example generation with KV cache:
        model.eval()
        tokens = model.generate(
            start_tokens,
            max_new_tokens=100,
            use_cache=True  # Dramatically faster
        )
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Main model components
        transformer_dict = dict(
            # Token embeddings: convert token IDs to vectors
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # Dropout
            drop = nn.Dropout(config.dropout),

            # Transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # Final layer normalization
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        )

        # Only add learned positional embeddings if not using RoPE
        if not config.use_rope:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(transformer_dict)

        # Language modeling head: project embeddings to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing between token embeddings and lm_head (like GPT-2/3/4)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters and optimizations
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"Using Flash Attention (scaled_dot_product_attention)")
        print(f"KV caching enabled for inference (O(n) generation)")
        if config.use_rope:
            print(f"Using Rotary Position Embeddings (RoPE)")
        if config.use_amp:
            print(f"Mixed Precision Training (AMP) enabled")
        if config.use_compile:
            print(f"torch.compile enabled for MLP fusion")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), subtract position and token embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            # Only subtract wpe if it exists (not using RoPE)
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights using GPT-2/3 initialization scheme."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        """
        Forward pass of the GPT model with optional KV caching.

        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices of shape (B, T) for training
            past_kvs: List of past key-value tuples for each layer (for generation)
            use_cache: If True, return present key-values for caching

        Returns:
            If targets is None: (logits, loss, present_kvs)
            If targets is provided: (logits, loss, None)
        """
        device = idx.device
        b, t = idx.size()

        # Calculate position offset for cached inference
        past_length = 0 if past_kvs is None else past_kvs[0][0].size(2)

        assert t + past_length <= self.config.block_size, \
            f"Cannot forward sequence of length {t + past_length}, block size is only {self.config.block_size}"

        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, n_embd)

        # Add positional embeddings only if not using RoPE
        if self.config.use_rope:
            # RoPE is applied within the attention mechanism
            x = self.transformer.drop(tok_emb)
        else:
            # Generate position indices [0, 1, 2, ..., t-1] with offset for cache
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # position embeddings (T, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks with KV caching
        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                present_kvs.append(present_kv)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Language modeling head
        if targets is not None:
            # Training mode: compute full logits and loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference mode: only compute logits for the last token
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss, present_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """
        Generate new tokens autoregressively with KV caching for speed.

        KV caching reduces generation from O(n²) to O(n) by reusing computed
        key-value pairs from previous tokens.

        Args:
            idx: Input token indices of shape (B, T)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
            use_cache: Use KV caching for faster generation (default: True)

        Returns:
            Generated token indices of shape (B, T + max_new_tokens)
        """
        past_kvs = None

        for _ in range(max_new_tokens):
            # With KV cache: only pass the new token
            # Without cache: pass full sequence (cropped to block_size)
            if use_cache and past_kvs is not None:
                idx_cond = idx[:, -1:]  # Only the last token
            else:
                # Crop context if it exceeds block size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass with caching
            logits, _, present_kvs = self(idx_cond, past_kvs=past_kvs, use_cache=use_cache)

            # Update cache for next iteration
            if use_cache:
                past_kvs = present_kvs

            # Get logits for last token and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def estimate_kv_cache_size(self, batch_size=1, seq_len=None):
        """
        Estimate the memory usage of KV cache.

        Args:
            batch_size: Batch size for generation
            seq_len: Sequence length (defaults to block_size)

        Returns:
            dict with cache size information
        """
        if seq_len is None:
            seq_len = self.config.block_size

        # Each layer stores K and V
        # Shape: (batch, n_head, seq_len, head_dim)
        head_dim = self.config.n_embd // self.config.n_head
        elements_per_layer = 2 * batch_size * self.config.n_head * seq_len * head_dim

        # Total elements across all layers
        total_elements = elements_per_layer * self.config.n_layer

        # Size in bytes (assuming float16 for inference)
        bytes_fp16 = total_elements * 2
        bytes_fp32 = total_elements * 4

        return {
            'elements_per_layer': elements_per_layer,
            'total_elements': total_elements,
            'size_mb_fp16': bytes_fp16 / (1024 ** 2),
            'size_mb_fp32': bytes_fp32 / (1024 ** 2),
            'size_gb_fp16': bytes_fp16 / (1024 ** 3),
            'size_gb_fp32': bytes_fp32 / (1024 ** 3),
        }

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure AdamW optimizer with weight decay only for 2D parameters.

        This is the optimization scheme used by GPT-2/3/4.
        """
        # Start with all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate into parameters that will have weight decay and those that won't
        # Weight decay is only applied to 2D parameters (weights), not 1D (biases, LayerNorm)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"Optimizer groups: {len(decay_params)} tensors with decay ({num_decay_params:,} params)")
        print(f"                  {len(nodecay_params)} tensors without decay ({num_nodecay_params:,} params)")

        # Create AdamW optimizer
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )

        return optimizer

    def get_grad_scaler(self, device_type):
        """
        Create a GradScaler for Automatic Mixed Precision training.

        Usage in training loop:
            scaler = model.get_grad_scaler(device_type)

            # In training step:
            with torch.amp.autocast(device_type='cuda', enabled=config.use_amp):
                logits, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        Args:
            device_type: 'cuda' or 'cpu'

        Returns:
            GradScaler if AMP is enabled and device is cuda, otherwise None
        """
        if self.config.use_amp and device_type == 'cuda':
            return torch.amp.GradScaler('cuda')
        return None


def create_gpt_mini(vocab_size: int = 100277, **kwargs) -> GPT:
    """
    Create a small GPT model for experimentation.

    Args:
        vocab_size: Vocabulary size (default: GPT-4 tokenizer size)
        **kwargs: Additional config parameters to override defaults

    Returns:
        GPT model instance
    """
    config = GPTConfig(vocab_size=vocab_size, **kwargs)
    model = GPT(config)
    return model


def create_gpt_medium(vocab_size: int = 100277, **kwargs) -> GPT:
    """
    Create a medium-sized GPT model (similar to GPT-2 medium).

    ~350M parameters
    """
    config_dict = dict(
        vocab_size=vocab_size,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        block_size=2048,
        dropout=0.1,
    )
    config_dict.update(kwargs)
    config = GPTConfig(**config_dict)
    model = GPT(config)
    return model


def create_gpt_large(vocab_size: int = 100277, **kwargs) -> GPT:
    """
    Create a large GPT model (similar to GPT-2 large).

    ~774M parameters
    """
    config_dict = dict(
        vocab_size=vocab_size,
        n_layer=36,
        n_head=20,
        n_embd=1280,
        block_size=2048,
        dropout=0.1,
    )
    config_dict.update(kwargs)
    config = GPTConfig(**config_dict)
    model = GPT(config)
    return model


if __name__ == "__main__":
    # Test the model
    print("="*70)
    print("GPT-4 Mini Model Test")
    print("="*70)

    # Create a small model for testing
    config = GPTConfig(
        vocab_size=100277,
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        dropout=0.1,
    )

    model = GPT(config)

    print(f"\nModel configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Context length: {config.block_size}")
    print(f"  Vocabulary size: {config.vocab_size:,}")

    # Test forward pass
    batch_size = 4
    seq_length = 64

    # Random input tokens
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    print(f"\nTesting forward pass...")
    print(f"  Input shape: {idx.shape}")

    logits, loss = model(idx, targets)

    print(f"  Output logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    print(f"\nTesting generation...")
    model.eval()

    # Start with a single token
    start_tokens = torch.tensor([[42]])  # arbitrary token
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8)

    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0].tolist()}")

    print("\n" + "="*70)
    print("Model test complete!")
    print("="*70)
