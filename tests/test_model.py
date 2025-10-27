"""Tests for model module."""

import pytest
import torch
from yagpt.model import GPT, GPTConfig, create_gpt_mini


def test_gpt_config():
    """Test GPTConfig creation and validation."""
    config = GPTConfig(
        vocab_size=100277,
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=2048,
    )
    assert config.n_embd % config.n_head == 0


def test_create_gpt_mini():
    """Test creating a GPT mini model."""
    model = create_gpt_mini(n_layer=2, n_head=2, n_embd=128, block_size=128)
    assert model is not None
    assert isinstance(model, GPT)


def test_model_forward():
    """Test model forward pass."""
    model = create_gpt_mini(n_layer=2, n_head=2, n_embd=128, block_size=128)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, 100277, (batch_size, seq_len))

    # Forward pass without targets
    logits, loss, _ = model(idx)

    assert logits.shape == (batch_size, 1, 100277)
    assert loss is None


def test_model_forward_with_targets():
    """Test model forward pass with targets."""
    model = create_gpt_mini(n_layer=2, n_head=2, n_embd=128, block_size=128)
    model.eval()

    # Create dummy input and targets
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, 100277, (batch_size, seq_len))
    targets = torch.randint(0, 100277, (batch_size, seq_len))

    # Forward pass with targets
    logits, loss, _ = model(idx, targets)

    assert logits.shape == (batch_size, seq_len, 100277)
    assert loss is not None
    assert isinstance(loss.item(), float)


def test_model_generation():
    """Test model text generation."""
    model = create_gpt_mini(n_layer=2, n_head=2, n_embd=128, block_size=128)
    model.eval()

    # Start with a single token
    start_tokens = torch.tensor([[42]])

    # Generate
    with torch.no_grad():
        generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8)

    assert generated.shape == (1, 11)  # 1 start token + 10 new tokens
