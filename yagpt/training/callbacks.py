"""
Training Callbacks - Modular hooks for the training loop.

Callbacks handle logging, checkpointing, evaluation, and other
periodic operations without cluttering the main training loop.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .trainer import Trainer


@dataclass
class TrainState:
    """Current training state passed to callbacks."""

    step: int
    loss: float
    lr: float
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    val_loss: float | None = None


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_step_end(self, trainer: "Trainer", state: TrainState) -> None:
        """Called at the end of each training step."""
        pass

    def on_eval_end(self, trainer: "Trainer", state: TrainState) -> None:
        """Called after evaluation."""
        pass


class LoggingCallback(Callback):
    """Log training metrics to console."""

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.start_time = None
        self.last_log_time = None
        self.last_log_step = 0

    def on_train_start(self, trainer: "Trainer") -> None:
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def on_step_end(self, trainer: "Trainer", state: TrainState) -> None:
        if state.step % self.log_interval != 0:
            return

        now = time.time()
        steps_since_last = state.step - self.last_log_step
        time_since_last = now - self.last_log_time

        tokens_per_sec = (
            steps_since_last * trainer.config.total_batch_size / time_since_last
            if time_since_last > 0 else 0
        )

        grad_str = f"grad={state.grad_norm:.2f}" if state.grad_norm else ""
        print(
            f"step {state.step:6d} | "
            f"loss {state.loss:.4f} | "
            f"lr {state.lr:.2e} | "
            f"{grad_str} | "
            f"{tokens_per_sec/1e6:.2f}M tok/s"
        )

        self.last_log_time = now
        self.last_log_step = state.step

    def on_eval_end(self, trainer: "Trainer", state: TrainState) -> None:
        print(f"step {state.step:6d} | val_loss {state.val_loss:.4f}")


class WandbCallback(Callback):
    """Log metrics to Weights & Biases."""

    def __init__(self, project: str, run_name: str | None = None):
        self.project = project
        self.run_name = run_name
        self._run = None

    def on_train_start(self, trainer: "Trainer") -> None:
        import wandb

        self._run = wandb.init(
            project=self.project,
            name=self.run_name,
            config=trainer.config.to_dict(),
        )

    def on_step_end(self, trainer: "Trainer", state: TrainState) -> None:
        if self._run is None:
            return

        metrics = {
            "train/loss": state.loss,
            "train/lr": state.lr,
        }
        if state.grad_norm is not None:
            metrics["train/grad_norm"] = state.grad_norm
        if state.tokens_per_sec is not None:
            metrics["train/tokens_per_sec"] = state.tokens_per_sec

        self._run.log(metrics, step=state.step)

    def on_eval_end(self, trainer: "Trainer", state: TrainState) -> None:
        if self._run is None or state.val_loss is None:
            return
        self._run.log({"val/loss": state.val_loss}, step=state.step)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._run is not None:
            self._run.finish()


class CheckpointCallback(Callback):
    """Save model checkpoints periodically."""

    def __init__(
        self,
        checkpoint_dir: str,
        interval: int = 1000,
        keep_last: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval = interval
        self.keep_last = keep_last
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: "Trainer", state: TrainState) -> None:
        if state.step == 0 or state.step % self.interval != 0:
            return
        self._save_checkpoint(trainer, state)
        self._cleanup_old_checkpoints()

    def on_train_end(self, trainer: "Trainer") -> None:
        # Save final checkpoint
        state = TrainState(
            step=trainer.step,
            loss=trainer.last_loss,
            lr=trainer.current_lr,
        )
        self._save_checkpoint(trainer, state, final=True)

    def _save_checkpoint(self, trainer: "Trainer", state: TrainState, final: bool = False):
        name = "final.pt" if final else f"step_{state.step:07d}.pt"
        path = self.checkpoint_dir / name

        checkpoint = {
            "step": state.step,
            "model": trainer.model.state_dict(),
            "optimizer": [opt.state_dict() for opt in trainer.optimizers],
            "config": trainer.config.to_dict(),
            "loss": state.loss,
        }
        if state.val_loss is not None:
            checkpoint["val_loss"] = state.val_loss

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for ckpt in checkpoints[:-self.keep_last]:
            ckpt.unlink()


class EvalCallback(Callback):
    """Run evaluation periodically."""

    def __init__(self, eval_interval: int = 1000, eval_steps: int = 50):
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps

    def on_step_end(self, trainer: "Trainer", state: TrainState) -> None:
        if state.step == 0 or state.step % self.eval_interval != 0:
            return

        val_loss = trainer.evaluate(self.eval_steps)
        state.val_loss = val_loss

        # Notify other callbacks
        for callback in trainer.callbacks:
            if callback is not self:
                callback.on_eval_end(trainer, state)


class SampleCallback(Callback):
    """Generate text samples periodically."""

    def __init__(
        self,
        interval: int = 1000,
        prompts: list[str] | None = None,
        max_tokens: int = 100,
        temperature: float = 0.8,
    ):
        self.interval = interval
        self.prompts = prompts or [
            "The future of artificial intelligence is",
            "Once upon a time,",
            "The key to success is",
        ]
        self.max_tokens = max_tokens
        self.temperature = temperature

    def on_step_end(self, trainer: "Trainer", state: TrainState) -> None:
        if state.step == 0 or state.step % self.interval != 0:
            return

        print(f"\n{'='*60}")
        print(f"Samples at step {state.step}")
        print('='*60)

        trainer.model.eval()
        for prompt in self.prompts:
            tokens = trainer.tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], device=trainer.config.device)

            output = trainer.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            text = trainer.tokenizer.decode(output[0].tolist())
            print(f"\n[Prompt] {prompt}")
            print(f"[Output] {text}")

        print('='*60 + "\n")
        trainer.model.train()
