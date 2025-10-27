"""Tests for tokenizer module."""

import pytest
from yagpt.tokenizer import GPT4Tokenizer, get_tokenizer


def test_gpt4_tokenizer_creation():
    """Test GPT4Tokenizer creation."""
    tokenizer = GPT4Tokenizer()
    assert tokenizer is not None
    assert tokenizer.vocab_size == 100277


def test_tokenizer_encode():
    """Test encoding text to tokens."""
    tokenizer = GPT4Tokenizer()
    text = "Hello, world!"
    tokens = tokenizer.encode(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_tokenizer_decode():
    """Test decoding tokens to text."""
    tokenizer = GPT4Tokenizer()
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_tokenizer_call():
    """Test tokenizer __call__ method."""
    tokenizer = GPT4Tokenizer()
    text = "Hello, world!"
    tokens = tokenizer(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_get_tokenizer_gpt4():
    """Test get_tokenizer for GPT-4."""
    tokenizer = get_tokenizer("gpt4")
    assert tokenizer is not None
    assert tokenizer.vocab_size == 100277


def test_get_tokenizer_gpt2():
    """Test get_tokenizer for GPT-2."""
    tokenizer = get_tokenizer("gpt2")
    assert tokenizer is not None
    assert tokenizer.vocab_size == 50257
