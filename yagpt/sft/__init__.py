"""
YAGPT Supervised Fine-Tuning - Chat-format SFT with loss masking.
"""

from .chat_format import format_chat, FormattedSample
from .config import SFTConfig
from .dataset import SFTDataset, sft_collate_fn, create_sft_dataloader
from .trainer import SFTTrainer

__all__ = [
    "format_chat",
    "FormattedSample",
    "SFTConfig",
    "SFTDataset",
    "sft_collate_fn",
    "create_sft_dataloader",
    "SFTTrainer",
]
