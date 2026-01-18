"""
Training Configuration - Flat, simple, explicit.

All training parameters in one place with sensible defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import yaml


@dataclass
class TrainConfig:
    """
    Training configuration.

    All parameters are flat (no nesting) for simplicity.
    Use TrainConfig.from_yaml() to load from file.
    """

    # === Model ===
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None  # None = same as n_heads (MHA)
    dim: int = 768
    hidden_dim: int | None = None  # None = 4 * dim
    max_seq_len: int = 2048

    # === Data ===
    train_data_dir: str = "./data/train"
    val_data_dir: str = "./data/val"
    tokenizer: str = "gpt2"

    # === Training ===
    batch_size: int = 32  # Micro-batch size per forward pass
    total_batch_size: int = 524288  # Total tokens per optimizer step
    max_steps: int = 100000

    # === Optimizer ===
    optimizer: Literal["adamw", "muon", "dual"] = "dual"  # dual = Muon + AdamW
    learning_rate: float = 3e-4  # Peak LR for AdamW / embeddings
    muon_lr: float = 0.02  # LR for Muon optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # === LR Schedule ===
    lr_schedule: Literal["warmup_cosine", "cosine", "three_phase"] = "warmup_cosine"
    warmup_ratio: float = 0.01  # Fraction of training for warmup
    min_lr_ratio: float = 0.1  # min_lr = learning_rate * min_lr_ratio

    # === Logging ===
    log_interval: int = 10
    eval_interval: int = 1000
    eval_steps: int = 50
    sample_interval: int = 1000  # Generate samples every N steps
    use_wandb: bool = False
    wandb_project: str = "yagpt"
    wandb_run_name: str | None = None

    # === Checkpointing ===
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000
    keep_checkpoints: int = 5  # Number of recent checkpoints to keep
    resume_from: str | None = None  # Path to checkpoint to resume from

    # === System ===
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    compile: bool = True  # Use torch.compile
    num_workers: int = 4  # DataLoader workers
    seed: int = 42

    # === Computed (set in __post_init__) ===
    grad_accum_steps: int = field(init=False)
    warmup_steps: int = field(init=False)
    min_lr: float = field(init=False)
    torch_dtype: torch.dtype = field(init=False)

    def __post_init__(self):
        # Compute gradient accumulation steps
        tokens_per_batch = self.batch_size * self.max_seq_len
        self.grad_accum_steps = self.total_batch_size // tokens_per_batch

        if self.total_batch_size % tokens_per_batch != 0:
            raise ValueError(
                f"total_batch_size ({self.total_batch_size}) must be divisible by "
                f"batch_size * max_seq_len ({tokens_per_batch})"
            )

        # Compute LR schedule parameters
        self.warmup_steps = int(self.warmup_ratio * self.max_steps)
        self.min_lr = self.learning_rate * self.min_lr_ratio

        # Map dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map[self.dtype]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        # Get all fields except computed ones
        data = {
            k: v for k, v in self.__dict__.items()
            if k not in ("grad_accum_steps", "warmup_steps", "min_lr", "torch_dtype")
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary (for logging)."""
        return {
            k: str(v) if isinstance(v, (Path, torch.dtype)) else v
            for k, v in self.__dict__.items()
        }
