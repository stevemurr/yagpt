"""
SFT Dataset - Load and format chat conversations for fine-tuning.

Expects JSONL files with the format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from yagpt.tokenizer import Tokenizer

from .chat_format import FormattedSample, format_chat, setup_chat_tokenizer, IGNORE_INDEX


class SFTDataset(IterableDataset):
    """
    Dataset for supervised fine-tuning on chat conversations.

    Reads JSONL files containing message lists and formats them
    with ChatML + loss masking.

    Args:
        data_path: Path to JSONL file or directory of JSONL files
        tokenizer: Tokenizer (ChatML tokens will be added)
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 2048,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        # Setup ChatML tokens
        self.im_start_id, self.im_end_id = setup_chat_tokenizer(tokenizer)

        # Find data files
        path = Path(data_path)
        if path.is_file():
            self.files = [path]
        elif path.is_dir():
            self.files = sorted(path.glob("*.jsonl"))
        else:
            raise ValueError(f"Data path not found: {data_path}")

        if not self.files:
            raise ValueError(f"No JSONL files found in {data_path}")

    def __iter__(self) -> Iterator[FormattedSample]:
        for filepath in self.files:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    messages = data.get("messages", [])

                    sample = format_chat(
                        messages=messages,
                        tokenizer=self.tokenizer,
                        max_seq_len=self.max_seq_len,
                        im_start_id=self.im_start_id,
                        im_end_id=self.im_end_id,
                    )

                    if sample is not None:
                        yield sample


def sft_collate_fn(
    batch: list[FormattedSample],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate FormattedSamples into padded batches.

    Pads input_ids with 0 and labels with IGNORE_INDEX.

    Returns:
        Tuple of (input_ids, labels) tensors of shape (batch, max_len)
    """
    max_len = max(len(s.input_ids) for s in batch)

    input_ids = []
    labels = []

    for sample in batch:
        pad_len = max_len - len(sample.input_ids)
        input_ids.append(sample.input_ids + [0] * pad_len)
        labels.append(sample.labels + [IGNORE_INDEX] * pad_len)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def create_sft_dataloader(
    data_path: str,
    tokenizer: Tokenizer,
    max_seq_len: int = 2048,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for SFT training.

    Args:
        data_path: Path to JSONL file or directory
        tokenizer: Tokenizer
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        DataLoader yielding (input_ids, labels) batches
    """
    dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=sft_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
