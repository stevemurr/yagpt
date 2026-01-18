"""
Streaming DataLoader for Parquet Shards.

Efficiently loads tokenized text data from sharded parquet files.
Supports:
- Streaming (no full dataset in memory)
- Multi-worker parallel loading
- Automatic tokenization or pre-tokenized data
"""

import glob
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

import pyarrow.parquet as pq


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for parquet shards.

    Reads parquet files containing either:
    - 'text' column: Raw text (will be tokenized on-the-fly)
    - 'tokens' column: Pre-tokenized sequences (faster)
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        tokenizer=None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Directory containing parquet shards
            seq_len: Sequence length for each sample
            tokenizer: Tokenizer for encoding text (required if not pre-tokenized)
            shuffle: Whether to shuffle shard order
            seed: Random seed for shuffling
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed

        # Find all parquet files
        self.shard_paths = sorted(glob.glob(str(self.data_dir / "*.parquet")))
        if not self.shard_paths:
            raise ValueError(f"No parquet files found in {data_dir}")

        # Detect data format from first shard
        self._detect_format()

    def _detect_format(self):
        """Detect if data is pre-tokenized or raw text."""
        table = pq.read_table(self.shard_paths[0], columns=None)
        columns = table.column_names

        if "tokens" in columns:
            self.token_column = "tokens"
            self.text_column = None
            print(f"Found pre-tokenized data in '{self.token_column}' column")
        elif "text" in columns:
            self.text_column = "text"
            self.token_column = None
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for text data")
            print(f"Found text data in '{self.text_column}' column (will tokenize)")
        else:
            raise ValueError(f"No 'tokens' or 'text' column found. Columns: {columns}")

    def _get_shards_for_worker(self) -> list[str]:
        """Get shard paths for this worker (supports multi-worker loading)."""
        shards = self.shard_paths.copy()

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(shards)

        # Split across workers if using DataLoader with num_workers > 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shards = [
                s for i, s in enumerate(shards)
                if i % worker_info.num_workers == worker_info.id
            ]

        return shards

    def _read_shard(self, shard_path: str) -> Iterator[list[int]]:
        """Read a shard and yield token sequences."""
        table = pq.read_table(shard_path)

        if self.token_column:
            # Pre-tokenized data
            for batch in table.to_batches(max_chunksize=1000):
                tokens_col = batch[self.token_column]
                for i in range(len(tokens_col)):
                    tokens = tokens_col[i].as_py()
                    yield tokens
        else:
            # Raw text - tokenize on the fly
            for batch in table.to_batches(max_chunksize=1000):
                text_col = batch[self.text_column]
                for i in range(len(text_col)):
                    text = text_col[i].as_py()
                    tokens = self.tokenizer.encode(text)
                    yield tokens

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over sequences of fixed length."""
        shards = self._get_shards_for_worker()
        buffer = []

        for shard_path in shards:
            try:
                for tokens in self._read_shard(shard_path):
                    buffer.extend(tokens)

                    # Yield complete sequences
                    while len(buffer) >= self.seq_len + 1:
                        # +1 for target offset
                        sequence = buffer[:self.seq_len + 1]
                        buffer = buffer[self.seq_len + 1:]
                        yield torch.tensor(sequence, dtype=torch.long)

            except Exception as e:
                print(f"Error reading {shard_path}: {e}")
                continue


def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that creates (x, y) pairs for language modeling.

    Takes sequences of length seq_len+1 and splits into:
    - x: tokens[:-1] (input)
    - y: tokens[1:]  (target)
    """
    sequences = torch.stack(batch)
    x = sequences[:, :-1]
    y = sequences[:, 1:]
    return x, y


def create_dataloader(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    tokenizer=None,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader for training/evaluation.

    Args:
        data_dir: Directory containing parquet shards
        seq_len: Sequence length for each sample
        batch_size: Batch size
        tokenizer: Tokenizer (required if data is not pre-tokenized)
        shuffle: Whether to shuffle shard order
        num_workers: Number of worker processes
        seed: Random seed

    Returns:
        DataLoader yielding (x, y) batches
    """
    dataset = StreamingDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        tokenizer=tokenizer,
        shuffle=shuffle,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
