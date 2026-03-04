"""
SFT Trainer - Supervised fine-tuning training loop.

Separate from the pre-training Trainer:
- Epoch-based (1-3 epochs typically)
- AdamW only (no Muon for SFT)
- Loss masking via ignore_index=-100
- Lower learning rates
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yagpt.models import GPT, GPTConfig
from yagpt.optim import get_lr_scheduler
from yagpt.tokenizer import Tokenizer
from yagpt.training.callbacks import Callback, TrainState

from .config import SFTConfig


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer.

    Loads a pre-trained checkpoint and fine-tunes on chat data
    with loss masking (only assistant turns contribute to loss).
    """

    def __init__(
        self,
        config: SFTConfig,
        model: GPT,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        callbacks: list[Callback] | None = None,
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        self.tokenizer = Tokenizer(config.tokenizer)

        # AdamW only for SFT
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        self.step = 0
        self.epoch = 0
        self.last_loss = 0.0
        self.current_lr = config.learning_rate

        # Estimate total steps for LR schedule
        self._total_steps = self._estimate_total_steps()
        warmup_steps = int(config.warmup_ratio * self._total_steps)

        self.lr_schedule = get_lr_scheduler(
            schedule="warmup_cosine",
            max_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.1,
            total_steps=self._total_steps,
            warmup_steps=warmup_steps,
        )

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps from dataset and epochs."""
        # For IterableDataset, estimate from first pass
        # This is approximate; will be refined during training
        return self.config.epochs * 10000  # Conservative estimate

    def _update_lr(self) -> float:
        lr = self.lr_schedule(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, float | None]:
        """Execute one SFT training step."""
        self.model.train()
        input_ids = input_ids.to(self.config.device)
        labels = labels.to(self.config.device)

        # Forward with ignore_index=-100 for masked labels
        with torch.amp.autocast(device_type="cuda", dtype=self.config.torch_dtype):
            # For SFT: input is tokens[:-1], target is tokens[1:]
            # But our labels are already aligned, so we shift here
            logits, loss, _ = self.model(
                input_ids[:, :-1],
                targets=labels[:, 1:],
                ignore_index=-100,
            )

        loss.backward()

        grad_norm = None
        if self.config.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            ).item()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item(), grad_norm

    @torch.no_grad()
    def evaluate(self, num_steps: int) -> float:
        """Run evaluation on validation data."""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        count = 0

        val_iter = iter(self.val_loader)
        for _ in range(num_steps):
            try:
                input_ids, labels = next(val_iter)
            except StopIteration:
                break

            input_ids = input_ids.to(self.config.device)
            labels = labels.to(self.config.device)

            with torch.amp.autocast(device_type="cuda", dtype=self.config.torch_dtype):
                _, loss, _ = self.model(
                    input_ids[:, :-1],
                    targets=labels[:, 1:],
                    ignore_index=-100,
                )

            total_loss += loss.item()
            count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def save_checkpoint(self, path: str) -> None:
        """Save SFT checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.step,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "loss": self.last_loss,
        }, path)
        print(f"Saved SFT checkpoint: {path}")

    def train(self) -> None:
        """Run the SFT training loop."""
        print(f"\nStarting SFT training")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print()

        for callback in self.callbacks:
            callback.on_train_start(self)

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            print(f"\n--- Epoch {epoch + 1}/{self.config.epochs} ---")

            for input_ids, labels in self.train_loader:
                self.current_lr = self._update_lr()
                loss, grad_norm = self.train_step(input_ids, labels)
                self.last_loss = loss
                self.step += 1

                state = TrainState(
                    step=self.step,
                    loss=loss,
                    lr=self.current_lr,
                    grad_norm=grad_norm,
                )

                for callback in self.callbacks:
                    callback.on_step_end(self, state)

            # Save after each epoch if configured
            if self.config.save_every_epoch:
                ckpt_dir = Path(self.config.checkpoint_dir)
                self.save_checkpoint(str(ckpt_dir / f"epoch_{epoch + 1}.pt"))

        # Save final checkpoint
        ckpt_dir = Path(self.config.checkpoint_dir)
        self.save_checkpoint(str(ckpt_dir / "sft_final.pt"))

        for callback in self.callbacks:
            callback.on_train_end(self)

        print("\nSFT training complete!")
