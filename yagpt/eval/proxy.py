"""
Proxy Evaluation Metrics.

Quick, cheap metrics that correlate with downstream task performance,
useful for monitoring during training without running full benchmarks.
"""

import math

from yagpt.tokenizer import Tokenizer


def compute_bits_per_byte(
    val_loss: float,
    tokenizer: Tokenizer | None = None,
    bytes_per_token: float | None = None,
) -> float:
    """
    Convert validation loss (nats per token) to bits per byte.

    Bits-per-byte normalizes across tokenizers, making it possible to
    compare models with different vocabularies on the same scale.

    BPB = (val_loss / ln(2)) / bytes_per_token

    Args:
        val_loss: Cross-entropy loss in nats (natural log)
        tokenizer: Tokenizer to estimate bytes_per_token from.
                   If None, bytes_per_token must be provided.
        bytes_per_token: Average bytes per token. Estimated from
                         tokenizer if not provided.

    Returns:
        Bits per byte value (lower is better, typically 0.7-1.5 for LLMs)
    """
    if bytes_per_token is None:
        if tokenizer is None:
            raise ValueError("Either tokenizer or bytes_per_token must be provided")
        bytes_per_token = _estimate_bytes_per_token(tokenizer)

    bits_per_token = val_loss / math.log(2)
    return bits_per_token / bytes_per_token


def _estimate_bytes_per_token(tokenizer: Tokenizer, sample_size: int = 1000) -> float:
    """
    Estimate average bytes per token for a tokenizer.

    Uses a diverse sample of English text to estimate the ratio.
    """
    # Representative English text sample
    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In the beginning was the Word, and the Word was with God. "
        "It was the best of times, it was the worst of times. "
        "To be, or not to be, that is the question. "
        "All happy families are alike; every unhappy family is unhappy in its own way. "
    ) * 10  # Repeat for better estimate

    tokens = tokenizer.encode(sample_text)
    text_bytes = len(sample_text.encode("utf-8"))

    return text_bytes / len(tokens)
