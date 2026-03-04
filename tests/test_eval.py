"""Tests for evaluation components."""

import math

import pytest
import torch

from yagpt.eval.proxy import compute_bits_per_byte
from yagpt.tokenizer import Tokenizer


class TestBitsPerByte:
    def test_basic_computation(self):
        # With known bytes_per_token, check the formula
        val_loss = 3.0  # nats
        bpb = compute_bits_per_byte(val_loss, bytes_per_token=4.0)

        expected = (val_loss / math.log(2)) / 4.0
        assert abs(bpb - expected) < 1e-6

    def test_lower_loss_means_lower_bpb(self):
        bpb_high = compute_bits_per_byte(4.0, bytes_per_token=4.0)
        bpb_low = compute_bits_per_byte(2.0, bytes_per_token=4.0)
        assert bpb_low < bpb_high

    def test_with_tokenizer(self):
        tokenizer = Tokenizer("gpt2")
        bpb = compute_bits_per_byte(3.0, tokenizer=tokenizer)
        # Should be a reasonable value
        assert 0.1 < bpb < 10.0

    def test_requires_tokenizer_or_bytes(self):
        with pytest.raises(ValueError):
            compute_bits_per_byte(3.0)


class TestYAGPTWrapper:
    """Test the lm-eval wrapper (without requiring lm-eval installed)."""

    def test_import_guard(self):
        # Should be importable even without lm-eval
        from yagpt.eval.harness import HAS_LM_EVAL
        # Just verify the module loads without error
        assert isinstance(HAS_LM_EVAL, bool)
