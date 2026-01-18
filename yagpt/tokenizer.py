"""
Tokenizer - Thin wrapper around tiktoken.

Provides a simple interface to tiktoken encodings used by GPT models.
"""

import tiktoken


class Tokenizer:
    """
    Tokenizer using tiktoken encodings.

    Supported encodings:
    - "gpt2" (r50k_base): 50,257 tokens, used by GPT-2
    - "gpt4" (cl100k_base): 100,277 tokens, used by GPT-4 and ChatGPT
    - "o200k_base": 200,019 tokens, used by GPT-4o
    """

    # Map friendly names to tiktoken encoding names
    ENCODING_MAP = {
        "gpt2": "r50k_base",
        "gpt4": "cl100k_base",
        "gpt4o": "o200k_base",
    }

    def __init__(self, encoding: str = "gpt2"):
        """
        Initialize tokenizer.

        Args:
            encoding: Encoding name ("gpt2", "gpt4", "gpt4o") or tiktoken name
        """
        encoding_name = self.ENCODING_MAP.get(encoding, encoding)
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = encoding

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self._encoding.n_vocab

    @property
    def eot_token(self) -> int:
        """End of text token ID."""
        return self._encoding.eot_token

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (not used, for API compat)

        Returns:
            List of token IDs
        """
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        return self._encoding.decode(token_ids)

    def __call__(self, text: str) -> list[int]:
        """Encode text (callable interface)."""
        return self.encode(text)

    def __repr__(self) -> str:
        return f"Tokenizer(encoding='{self.name}', vocab_size={self.vocab_size})"
