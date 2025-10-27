"""
FineWeb Dataset Loader for GPT Pre-training

This module implements an IterableDataset for loading sharded parquet files
from the FineWeb dataset for GPT-style language model pre-training.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import glob
import os
from pathlib import Path

try:
    import pyarrow.parquet as pq
    BACKEND = 'pyarrow'
except ImportError:
    try:
        import pandas as pd
        BACKEND = 'pandas'
    except ImportError:
        raise ImportError("Either pyarrow or pandas is required to read parquet files")


class FineWebDataset(IterableDataset):
    """
    Streaming dataset for FineWeb parquet shards.

    Args:
        data_dir: Directory containing parquet shard files
        seq_length: Maximum sequence length for each sample
        tokenizer: Optional tokenizer callable (if None, assumes pre-tokenized data)
        shuffle_shards: Whether to shuffle shard order
        text_column: Name of the text column in parquet files (default: 'text')
        token_column: Name of pre-tokenized token column (if exists)
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 2048,
        tokenizer=None,
        shuffle_shards: bool = True,
        text_column: str = 'text',
        token_column: str = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.shuffle_shards = shuffle_shards
        self.text_column = text_column
        self.token_column = token_column

        # Find all parquet files
        self.shard_paths = sorted(glob.glob(str(self.data_dir / "*.parquet")))
        if len(self.shard_paths) == 0:
            raise ValueError(f"No parquet files found in {data_dir}")

        print(f"Found {len(self.shard_paths)} shards in {data_dir}")

        # For distributed training support
        self.rank = 0
        self.world_size = 1

    def set_distributed_info(self, rank: int, world_size: int):
        """Set rank and world_size for distributed training."""
        self.rank = rank
        self.world_size = world_size

    def _get_shard_paths_for_worker(self):
        """Get shard paths assigned to this worker."""
        worker_info = torch.utils.data.get_worker_info()
        shard_paths = self.shard_paths.copy()

        if self.shuffle_shards:
            # Use a deterministic shuffle based on epoch/worker
            # For now, simple shuffle - can be made deterministic later
            import random
            random.shuffle(shard_paths)

        # Distribute shards across distributed ranks
        shards_per_rank = len(shard_paths) // self.world_size
        rank_start = self.rank * shards_per_rank
        rank_end = rank_start + shards_per_rank if self.rank < self.world_size - 1 else len(shard_paths)
        shard_paths = shard_paths[rank_start:rank_end]

        # Further split among workers if using DataLoader with num_workers > 0
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            # Assign shards to this worker
            worker_shards = [
                shard for i, shard in enumerate(shard_paths)
                if i % num_workers == worker_id
            ]
            return worker_shards

        return shard_paths

    def _read_parquet_shard(self, shard_path: str):
        """Read a parquet shard and yield rows."""
        if BACKEND == 'pyarrow':
            table = pq.read_table(shard_path)
            # Convert to batches for memory efficiency
            for batch in table.to_batches(max_chunksize=1000):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    yield row
        else:  # pandas
            df = pd.read_parquet(shard_path)
            for _, row in df.iterrows():
                yield row

    def _tokenize_text(self, text: str):
        """Tokenize text if tokenizer is provided."""
        if self.tokenizer is not None:
            return self.tokenizer(text)
        else:
            # No tokenizer provided - raise error with helpful message
            raise ValueError(
                "No tokenizer provided. You must pass a tokenizer function that "
                "converts text to a list of integer token IDs. "
                "Example: tokenizer=lambda text: [ord(c) for c in text]"
            )

    def _create_sequences(self, tokens, seq_length: int):
        """
        Create fixed-length sequences from a stream of tokens.
        Yields sequences of exactly seq_length tokens.
        """
        buffer = []

        if isinstance(tokens, (list, tuple)):
            buffer.extend(tokens)
        elif isinstance(tokens, torch.Tensor):
            buffer.extend(tokens.tolist())
        elif isinstance(tokens, str):
            # If tokens is still text, split by whitespace as fallback
            buffer.extend(tokens.split())

        # Yield complete sequences
        while len(buffer) >= seq_length:
            sequence = buffer[:seq_length]
            buffer = buffer[seq_length:]
            yield torch.tensor(sequence, dtype=torch.long)

        # Note: Remaining tokens in buffer are dropped
        # For proper training, you'd want to carry them over to next document
        # or handle them differently

    def __iter__(self):
        """Iterate over the dataset, yielding sequences of tokens."""
        shard_paths = self._get_shard_paths_for_worker()

        token_buffer = []

        for shard_path in shard_paths:
            try:
                for row in self._read_parquet_shard(shard_path):
                    # Extract tokens from row
                    if self.token_column and self.token_column in row:
                        # Use pre-tokenized data
                        tokens = row[self.token_column]
                    elif self.text_column in row:
                        # Tokenize text
                        text = row[self.text_column]
                        tokens = self._tokenize_text(text)
                    else:
                        # Skip row if no valid column found
                        continue

                    # Add tokens to buffer
                    if isinstance(tokens, (list, tuple)):
                        # Ensure all tokens are integers
                        for tok in tokens:
                            if not isinstance(tok, (int, float)):
                                raise ValueError(
                                    f"Tokenizer must return list of integers, got {type(tok)}. "
                                    f"Token value: {tok}"
                                )
                        token_buffer.extend(tokens)
                    elif isinstance(tokens, torch.Tensor):
                        token_buffer.extend(tokens.tolist())
                    else:
                        raise ValueError(
                            f"Tokenizer returned unexpected type: {type(tokens)}. "
                            "Must return a list of integers or torch.Tensor."
                        )

                    # Yield sequences when buffer is large enough
                    while len(token_buffer) >= self.seq_length:
                        sequence = token_buffer[:self.seq_length]
                        token_buffer = token_buffer[self.seq_length:]
                        yield torch.tensor(sequence, dtype=torch.long)

            except Exception as e:
                print(f"Error reading shard {shard_path}: {e}")
                continue


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    seq_length: int = 2048,
    num_workers: int = 4,
    tokenizer=None,
    shuffle_shards: bool = True,
    **kwargs
):
    """
    Create a DataLoader for FineWeb dataset.

    Args:
        data_dir: Directory containing parquet shards
        batch_size: Batch size
        seq_length: Sequence length for each sample
        num_workers: Number of worker processes for data loading
        tokenizer: Optional tokenizer callable
        shuffle_shards: Whether to shuffle shard order
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        DataLoader instance
    """
    dataset = FineWebDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        tokenizer=tokenizer,
        shuffle_shards=shuffle_shards,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        **kwargs
    )

    return dataloader


if __name__ == "__main__":
    # Example usage
    print(f"Using backend: {BACKEND}")

    # Import GPT-4 tokenizer
    from gpt_mini.tokenizer import GPT4Tokenizer

    tokenizer = GPT4Tokenizer()
    print(f"Using GPT-4 tokenizer (vocab size: {tokenizer.vocab_size:,})")

    # Create dataset
    dataset = FineWebDataset(
        data_dir="./datasets/fineweb",
        seq_length=128,  # Small for testing
        shuffle_shards=False,
        tokenizer=tokenizer,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,  # 0 for debugging, increase for production
    )

    # Test iteration
    print("\nTesting dataloader...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: shape={batch.shape}, dtype={batch.dtype}")
        print(f"Sample data: {batch[0][:20]}")  # First 20 tokens of first sequence

        # Decode first sequence to verify
        if i == 0:
            decoded = tokenizer.decode(batch[0].tolist())
            print(f"Decoded sample: {decoded[:100]}...")

        if i >= 2:  # Just test a few batches
            break

    print("\nDataloader test complete!")
