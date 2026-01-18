"""
Learning Rate Schedules.

Provides common LR schedules for training transformers:
- Warmup + Cosine decay (most common)
- Warmup + Constant + Linear decay (3-phase)
"""

import math
from dataclasses import dataclass


@dataclass
class CosineSchedule:
    """Simple cosine annealing schedule."""

    max_lr: float
    min_lr: float
    total_steps: int

    def __call__(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        progress = step / self.total_steps
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


@dataclass
class WarmupCosineSchedule:
    """
    Warmup + Cosine decay schedule.

    The most common schedule for training transformers:
    1. Linear warmup from 0 to max_lr
    2. Cosine decay from max_lr to min_lr
    """

    max_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int

    def __call__(self, step: int) -> float:
        # Warmup phase
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # Cosine decay phase
        if step >= self.total_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


@dataclass
class ThreePhaseSchedule:
    """
    Three-phase schedule: Warmup + Constant + Decay.

    Used in some training runs where you want a longer period at peak LR:
    1. Linear warmup (e.g., 2% of training)
    2. Constant at max_lr (e.g., 78% of training)
    3. Cosine decay to min_lr (e.g., 20% of training)
    """

    max_lr: float
    min_lr: float
    warmup_steps: int
    stable_steps: int  # Steps at constant LR after warmup
    decay_steps: int

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.stable_steps + self.decay_steps

    def __call__(self, step: int) -> float:
        # Phase 1: Warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # Phase 2: Stable
        step_after_warmup = step - self.warmup_steps
        if step_after_warmup < self.stable_steps:
            return self.max_lr

        # Phase 3: Decay
        step_in_decay = step_after_warmup - self.stable_steps
        if step_in_decay >= self.decay_steps:
            return self.min_lr

        progress = step_in_decay / self.decay_steps
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


def get_lr_scheduler(
    schedule: str,
    max_lr: float,
    min_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    **kwargs,
):
    """
    Factory function to create LR schedulers.

    Args:
        schedule: One of "cosine", "warmup_cosine", "three_phase"
        max_lr: Peak learning rate
        min_lr: Minimum learning rate
        total_steps: Total training steps
        warmup_steps: Number of warmup steps (for warmup schedules)
        **kwargs: Additional arguments for specific schedules

    Returns:
        Callable that takes step and returns learning rate
    """
    if schedule == "cosine":
        return CosineSchedule(max_lr=max_lr, min_lr=min_lr, total_steps=total_steps)

    elif schedule == "warmup_cosine":
        return WarmupCosineSchedule(
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

    elif schedule == "three_phase":
        stable_steps = kwargs.get("stable_steps", total_steps - warmup_steps - total_steps // 5)
        decay_steps = total_steps - warmup_steps - stable_steps
        return ThreePhaseSchedule(
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
        )

    else:
        raise ValueError(f"Unknown schedule: {schedule}")
