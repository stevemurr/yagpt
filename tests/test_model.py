"""Tests for the GPT model."""

import pytest
import torch

from yagpt.models import GPT, GPTConfig
from yagpt.models.attention import CausalAttention
from yagpt.models.mlp import SwiGLU
from yagpt.models.norm import RMSNorm
from yagpt.models.rope import apply_rope, build_rope_cache


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_output(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        # RMS should be approximately 1 after normalization
        rms = (out ** 2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestRoPE:
    def test_cache_shape(self):
        cos, sin = build_rope_cache(seq_len=128, head_dim=64)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_apply_rope_shape(self):
        batch, seq, heads, dim = 2, 16, 8, 64
        x = torch.randn(batch, seq, heads, dim)
        cos, sin = build_rope_cache(seq_len=seq, head_dim=dim)

        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_apply_rope_dtype(self):
        x = torch.randn(2, 16, 8, 64, dtype=torch.bfloat16)
        cos, sin = build_rope_cache(seq_len=16, head_dim=64)
        out = apply_rope(x, cos, sin)
        assert out.dtype == torch.bfloat16


class TestSwiGLU:
    def test_output_shape(self):
        mlp = SwiGLU(dim=64, hidden_dim=256)
        x = torch.randn(2, 10, 64)
        out = mlp(x)
        assert out.shape == x.shape

    def test_default_hidden_dim(self):
        mlp = SwiGLU(dim=64)
        assert mlp.gate_up.out_features == 2 * 4 * 64


class TestCausalAttention:
    def test_output_shape(self):
        attn = CausalAttention(dim=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        cos, sin = build_rope_cache(seq_len=16, head_dim=16)

        out, _ = attn(x, cos, sin)
        assert out.shape == x.shape

    def test_gqa_output_shape(self):
        # Grouped Query Attention: 8 query heads, 2 KV heads
        attn = CausalAttention(dim=64, n_heads=8, n_kv_heads=2)
        x = torch.randn(2, 16, 64)
        cos, sin = build_rope_cache(seq_len=16, head_dim=8)

        out, _ = attn(x, cos, sin)
        assert out.shape == x.shape

    def test_kv_cache(self):
        attn = CausalAttention(dim=64, n_heads=4)
        cos, sin = build_rope_cache(seq_len=32, head_dim=16)

        # Initial forward with full sequence
        x1 = torch.randn(2, 16, 64)
        _, kv_cache = attn(x1, cos[:16], sin[:16])

        # Subsequent forward with single token
        x2 = torch.randn(2, 1, 64)
        out, new_cache = attn(x2, cos[16:17], sin[16:17], kv_cache)

        assert out.shape == (2, 1, 64)
        assert new_cache[0].shape[1] == 17  # 16 + 1 cached keys


class TestGPT:
    @pytest.fixture
    def small_config(self):
        return GPTConfig(
            vocab_size=1000,
            n_layers=2,
            n_heads=4,
            dim=64,
            max_seq_len=128,
        )

    def test_forward_no_targets(self, small_config):
        model = GPT(small_config)
        x = torch.randint(0, 1000, (2, 16))

        logits, loss, _ = model(x)

        # Without targets, only returns last token logits
        assert logits.shape == (2, 1, 1000)
        assert loss is None

    def test_forward_with_targets(self, small_config):
        model = GPT(small_config)
        x = torch.randint(0, 1000, (2, 16))
        y = torch.randint(0, 1000, (2, 16))

        logits, loss, _ = model(x, y)

        # With targets, returns full logits
        assert logits.shape == (2, 16, 1000)
        assert loss is not None
        assert loss.ndim == 0  # Scalar

    def test_generate(self, small_config):
        model = GPT(small_config)
        model.eval()

        x = torch.randint(0, 1000, (1, 4))

        with torch.inference_mode():
            output = model.generate(x, max_new_tokens=10)

        assert output.shape == (1, 14)  # 4 input + 10 generated

    def test_num_parameters(self, small_config):
        model = GPT(small_config)
        n_params = model.num_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_weight_tying(self, small_config):
        model = GPT(small_config)
        # Token embeddings should share weights with lm_head
        assert model.tok_emb.weight is model.lm_head.weight
