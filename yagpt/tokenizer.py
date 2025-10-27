"""
Tokenizer module for GPT-4 clone using tiktoken.

Uses the cl100k_base encoding which is used by GPT-4.
"""

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required for tokenization. Install it with:\n"
        "pip install tiktoken"
    )


class GPT4Tokenizer:
    """
    Wrapper around tiktoken's cl100k_base encoding (used by GPT-4).

    Vocabulary size: 100,277 tokens
    """

    def __init__(self):
        # cl100k_base is the encoding used by GPT-4 and GPT-3.5-turbo
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.

        Args:
            text: Input text string

        Returns:
            List of integer token IDs
        """
        return self.encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back into text.

        Args:
            token_ids: List of integer token IDs

        Returns:
            Decoded text string
        """
        return self.encoding.decode(token_ids)

    def __call__(self, text: str) -> list[int]:
        """Allow tokenizer to be called directly."""
        return self.encode(text)


def get_tokenizer(name: str = "gpt4"):
    """
    Get a tokenizer by name.

    Args:
        name: Tokenizer name. Options:
            - "gpt4" or "cl100k_base": GPT-4 encoding (100k vocab)
            - "gpt2" or "r50k_base": GPT-2 encoding (50k vocab)
            - "p50k_base": Code models encoding

    Returns:
        Tokenizer instance
    """
    encoding_map = {
        "gpt4": "cl100k_base",
        "gpt2": "r50k_base",
        "code": "p50k_base",
    }

    encoding_name = encoding_map.get(name, name)

    class Tokenizer:
        def __init__(self, encoding_name):
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.vocab_size = self.encoding.n_vocab

        def encode(self, text: str) -> list[int]:
            return self.encoding.encode(text)

        def decode(self, token_ids: list[int]) -> str:
            return self.encoding.decode(token_ids)

        def __call__(self, text: str) -> list[int]:
            return self.encode(text)

    return Tokenizer(encoding_name)


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = GPT4Tokenizer()

    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size:,}")

    # Test encoding
    text = "Hello, world! This is a test of the GPT-4 tokenizer."
    tokens = tokenizer.encode(text)

    print(f"\nOriginal text: {text}")
    print(f"Token IDs: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: {decoded}")
    print(f"Match: {text == decoded}")

    # Show some examples of tokenization
    examples = [
        "The quick brown fox",
        "GPT-4 is awesome!",
        "Machine learning 🤖",
        "def hello_world():\n    print('Hello!')",
    ]

    print("\n" + "="*60)
    print("Tokenization Examples:")
    print("="*60)

    for example in examples:
        tokens = tokenizer.encode(example)
        print(f"\nText: {example!r}")
        print(f"Tokens: {tokens}")
        print(f"Count: {len(tokens)}")
