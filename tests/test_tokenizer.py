"""Tests for the tokenizer."""

import pytest

from yagpt.tokenizer import Tokenizer


class TestTokenizer:
    def test_gpt2_tokenizer(self):
        tok = Tokenizer("gpt2")
        assert tok.vocab_size == 50257

    def test_gpt4_tokenizer(self):
        tok = Tokenizer("gpt4")
        assert tok.vocab_size == 100277

    def test_encode_decode_roundtrip(self):
        tok = Tokenizer("gpt2")
        text = "Hello, world!"

        tokens = tok.encode(text)
        decoded = tok.decode(tokens)

        assert decoded == text

    def test_encode_returns_list(self):
        tok = Tokenizer("gpt2")
        tokens = tok.encode("test")

        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_callable_interface(self):
        tok = Tokenizer("gpt2")

        tokens1 = tok.encode("test")
        tokens2 = tok("test")

        assert tokens1 == tokens2

    def test_repr(self):
        tok = Tokenizer("gpt2")
        repr_str = repr(tok)

        assert "gpt2" in repr_str
        assert "50257" in repr_str
