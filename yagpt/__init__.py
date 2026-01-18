"""
YAGPT - Yet Another GPT

A clean, modular GPT implementation for learning and experimentation.
"""

__version__ = "0.2.0"

# Models
from yagpt.models import GPT, GPTConfig

# Tokenizer
from yagpt.tokenizer import Tokenizer

# Training
from yagpt.training import Trainer, TrainConfig

# Optimizers
from yagpt.optim import Muon

# Data
from yagpt.data import create_dataloader, StreamingDataset

__all__ = [
    # Models
    "GPT",
    "GPTConfig",
    # Tokenizer
    "Tokenizer",
    # Training
    "Trainer",
    "TrainConfig",
    # Optimizers
    "Muon",
    # Data
    "create_dataloader",
    "StreamingDataset",
]
