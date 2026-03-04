"""Tests for LoRA and QLoRA components."""

import pytest
import torch
import torch.nn as nn

from yagpt.models import GPT, GPTConfig
from yagpt.lora.lora import LoRALinear, apply_lora, lora_state_dict, count_lora_params
from yagpt.lora.qlora import NF4Linear, quantize_model_nf4


class TestLoRALinear:
    def test_output_shape(self):
        base = nn.Linear(64, 128, bias=False)
        lora = LoRALinear(base, rank=8, alpha=16)

        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 128)

    def test_identity_at_init(self):
        """B initialized to zeros means LoRA starts as identity."""
        base = nn.Linear(64, 128, bias=False)
        x = torch.randn(2, 10, 64)

        base_out = base(x)
        lora = LoRALinear(base, rank=8)
        lora_out = lora(x)

        assert torch.allclose(base_out, lora_out, atol=1e-6)

    def test_base_frozen(self):
        base = nn.Linear(64, 128, bias=False)
        lora = LoRALinear(base, rank=8)

        assert not lora.base.weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_different_from_base_after_training(self):
        base = nn.Linear(64, 128, bias=False)
        lora = LoRALinear(base, rank=8)

        # Simulate training by modifying LoRA params
        with torch.no_grad():
            lora.lora_B.fill_(1.0)

        x = torch.randn(2, 10, 64)
        base_out = base(x)
        lora_out = lora(x)

        assert not torch.allclose(base_out, lora_out)


class TestApplyLoRA:
    @pytest.fixture
    def small_model(self):
        config = GPTConfig(vocab_size=1024, n_layers=2, n_heads=4, dim=64)
        return GPT(config)

    def test_apply_lora_to_gpt(self, small_model):
        apply_lora(small_model, rank=8)

        # Check that LoRA was applied
        lora_count = sum(
            1 for _, m in small_model.named_modules()
            if isinstance(m, LoRALinear)
        )
        assert lora_count > 0

    def test_lora_state_dict(self, small_model):
        apply_lora(small_model, rank=8)
        sd = lora_state_dict(small_model)

        assert len(sd) > 0
        assert all("lora_A" in k or "lora_B" in k for k in sd)

    def test_count_lora_params(self, small_model):
        apply_lora(small_model, rank=8)
        lora_params, total_params = count_lora_params(small_model)

        assert lora_params > 0
        assert lora_params < total_params

    def test_forward_after_lora(self, small_model):
        """Model should still produce valid output after LoRA."""
        apply_lora(small_model, rank=8)

        x = torch.randint(0, 1024, (2, 16))
        logits, _, _ = small_model(x)
        assert logits.shape == (2, 1, 1024)


class TestNF4Linear:
    def test_quantize_dequantize(self):
        """Quantize then dequantize should be close to original."""
        base = nn.Linear(64, 128, bias=False)
        nn.init.normal_(base.weight, std=0.02)

        nf4 = NF4Linear.from_linear(base)
        reconstructed = nf4.dequantize()

        # NF4 is lossy but should be reasonably close
        error = (base.weight.data - reconstructed).abs()
        relative_error = error.mean() / base.weight.data.abs().mean()
        assert relative_error < 0.15  # Less than 15% relative error

    def test_output_shape(self):
        base = nn.Linear(64, 128, bias=False)
        nf4 = NF4Linear.from_linear(base)

        x = torch.randn(2, 10, 64)
        out = nf4(x)
        assert out.shape == (2, 10, 128)

    def test_with_bias(self):
        base = nn.Linear(64, 128, bias=True)
        nf4 = NF4Linear.from_linear(base)

        x = torch.randn(2, 10, 64)
        out = nf4(x)
        assert out.shape == (2, 10, 128)


class TestQuantizeModel:
    def test_quantize_gpt(self):
        config = GPTConfig(vocab_size=1024, n_layers=2, n_heads=4, dim=64)
        model = GPT(config)

        quantize_model_nf4(model)

        # Check that Linear layers were replaced (except lm_head)
        nf4_count = sum(
            1 for _, m in model.named_modules()
            if isinstance(m, NF4Linear)
        )
        assert nf4_count > 0

        # lm_head should NOT be quantized
        assert isinstance(model.lm_head, nn.Linear)
        assert not isinstance(model.lm_head, NF4Linear)

    def test_forward_after_quantize(self):
        config = GPTConfig(vocab_size=1024, n_layers=2, n_heads=4, dim=64)
        model = GPT(config)
        quantize_model_nf4(model)

        x = torch.randint(0, 1024, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 1, 1024)
