"""WebSocketCallback — bridges yagpt training callbacks to WebSocket broadcasts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from yagpt.training.callbacks import Callback, TrainState

if TYPE_CHECKING:
    from .state import PipelineState


class WebSocketCallback(Callback):
    """Streams training metrics over WebSocket and handles stop requests."""

    def __init__(self, pipeline: PipelineState, stage: str) -> None:
        self.pipeline = pipeline
        self.stage = stage

    def on_train_start(self, trainer: object) -> None:
        self.pipeline.broadcast_status(self.stage, "running")

    def on_step_end(self, trainer: object, state: TrainState) -> None:
        # Check for user-requested stop — handle different trainer loop patterns
        if self.pipeline.stop_event.is_set():
            # Pretrain Trainer: checks self.step < self.config.max_steps
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'max_steps'):  # type: ignore[attr-defined]
                trainer.config.max_steps = trainer.step  # type: ignore[attr-defined]
            # Alignment trainers: check self.step < self.max_steps
            if hasattr(trainer, 'max_steps'):
                trainer.max_steps = trainer.step  # type: ignore[attr-defined]
            # SFT trainer: epoch-based loop, set epochs to 0 to break
            if hasattr(trainer, 'epoch') and hasattr(trainer, 'config'):
                trainer.config.epochs = trainer.epoch  # type: ignore[attr-defined]

        data = {
            "stage": self.stage,
            "step": state.step,
            "loss": state.loss,
            "lr": state.lr,
            "grad_norm": state.grad_norm,
            "tokens_per_sec": state.tokens_per_sec,
            "mfu": state.mfu,
        }
        self.pipeline.metrics[self.stage].append(data)
        self.pipeline.broadcast({"type": "train_step", "data": data})

    def on_eval_end(self, trainer: object, state: TrainState) -> None:
        self.pipeline.broadcast({
            "type": "eval_result",
            "data": {
                "stage": self.stage,
                "step": state.step,
                "val_loss": state.val_loss,
            },
        })

    def on_train_end(self, trainer: object) -> None:
        status = "error" if self.pipeline.stop_event.is_set() else "done"
        if self.pipeline.stop_event.is_set():
            status = "stopped"
        self.pipeline.broadcast_status(self.stage, status)
        self.pipeline.broadcast({"type": "train_complete", "data": {"stage": self.stage}})
