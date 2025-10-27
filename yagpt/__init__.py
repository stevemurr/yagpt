"""
YAGPT (Yet Another GPT): Educational GPT-style language model implementation in PyTorch.

This package provides a clean, well-documented implementation of a GPT-style
transformer model for educational purposes and experimentation.
"""

__version__ = "0.1.0"

from yagpt.model import GPT, GPTConfig, create_gpt_mini, create_gpt_medium, create_gpt_large
from yagpt.tokenizer import GPT4Tokenizer, get_tokenizer
from yagpt.train import train, TrainingConfig
from yagpt.generate import generate_text, load_model_from_checkpoint
from yagpt.logger import create_logger
from yagpt.dataloader import FineWebDataset

__all__ = [
    "GPT",
    "GPTConfig",
    "create_gpt_mini",
    "create_gpt_medium",
    "create_gpt_large",
    "GPT4Tokenizer",
    "get_tokenizer",
    "train",
    "TrainingConfig",
    "generate_text",
    "load_model_from_checkpoint",
    "create_logger",
    "FineWebDataset",
]
