"""
SFT Configuration - Flat config for supervised fine-tuning.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml


@dataclass
class SFTConfig:
    """
    Configuration for supervised fine-tuning.

    SFT uses lower learning rates, AdamW only (no Muon), and runs
    for a small number of epochs rather than step-based.
    """

    # === Checkpoint ===
    base_checkpoint: str = ""  # Path to pre-trained checkpoint

    # === Data ===
    data_path: str = "./data/sft"  # JSONL file or directory
    tokenizer: str = "gpt2"
    max_seq_len: int = 2048

    # === Training ===
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5  # Lower LR for fine-tuning
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # === Logging ===
    log_interval: int = 10
    eval_interval: int = 500
    eval_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "yagpt-sft"

    # === Checkpointing ===
    checkpoint_dir: str = "./checkpoints/sft"
    save_every_epoch: bool = True

    # === System ===
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    num_workers: int = 0
    seed: int = 42

    @property
    def torch_dtype(self) -> torch.dtype:
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return dtype_map[self.dtype]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert to dictionary (for logging)."""
        return {
            k: str(v) if isinstance(v, (Path, torch.dtype)) else v
            for k, v in self.__dict__.items()
        }
