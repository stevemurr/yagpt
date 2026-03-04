"""
Preference Dataset for alignment training.

Loads JSONL files with preference pairs:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from yagpt.tokenizer import Tokenizer


class PreferenceDataset(IterableDataset):
    """
    Dataset for preference-based alignment (DPO, SimPO, etc.).

    Each sample contains a prompt with a chosen and rejected completion.

    Args:
        data_path: Path to JSONL file or directory
        tokenizer: Tokenizer
        max_seq_len: Maximum sequence length for each completion
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        path = Path(data_path)
        if path.is_file():
            self.files = [path]
        elif path.is_dir():
            self.files = sorted(path.glob("*.jsonl"))
        else:
            raise ValueError(f"Data path not found: {data_path}")

        if not self.files:
            raise ValueError(f"No JSONL files found in {data_path}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for filepath in self.files:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    sample = self._tokenize(data)
                    if sample is not None:
                        yield sample

    def _tokenize(self, data: dict) -> dict[str, torch.Tensor] | None:
        prompt = data.get("prompt", "")
        chosen = data.get("chosen", "")
        rejected = data.get("rejected", "")

        if not prompt or not chosen or not rejected:
            return None

        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen)
        rejected_ids = self.tokenizer.encode(rejected)

        # Truncate if needed
        max_completion = self.max_seq_len - len(prompt_ids)
        if max_completion <= 0:
            prompt_ids = prompt_ids[:self.max_seq_len // 2]
            max_completion = self.max_seq_len - len(prompt_ids)

        chosen_ids = chosen_ids[:max_completion]
        rejected_ids = rejected_ids[:max_completion]

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "chosen_ids": torch.tensor(prompt_ids + chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(prompt_ids + rejected_ids, dtype=torch.long),
            "prompt_len": torch.tensor(len(prompt_ids), dtype=torch.long),
        }


def preference_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate preference samples with padding."""
    max_chosen = max(s["chosen_ids"].shape[0] for s in batch)
    max_rejected = max(s["rejected_ids"].shape[0] for s in batch)
    max_prompt = max(s["prompt_ids"].shape[0] for s in batch)

    chosen_ids = torch.zeros(len(batch), max_chosen, dtype=torch.long)
    rejected_ids = torch.zeros(len(batch), max_rejected, dtype=torch.long)
    prompt_ids = torch.zeros(len(batch), max_prompt, dtype=torch.long)
    prompt_lens = torch.stack([s["prompt_len"] for s in batch])

    for i, s in enumerate(batch):
        chosen_ids[i, :s["chosen_ids"].shape[0]] = s["chosen_ids"]
        rejected_ids[i, :s["rejected_ids"].shape[0]] = s["rejected_ids"]
        prompt_ids[i, :s["prompt_ids"].shape[0]] = s["prompt_ids"]

    return {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "prompt_len": prompt_lens,
    }
