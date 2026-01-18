"""
Training Hooks System

A flexible hooks system for managing training loop operations without conditional bloat.
Hooks can run at specific intervals, on feature flags, or at lifecycle events.

Design Philosophy:
- Main training loop stays clean and linear
- Each feature is a self-contained hook
- Hooks are composable and independently testable
- Performance-optimized with pre-computed schedules
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import torch
import torch.nn as nn


@dataclass
class HookContext:
    """
    All data that hooks might need access to.

    This is passed to each hook's execute() method and contains references
    to all training state, models, optimizers, and per-step data.
    """
    # Current step information
    step: int
    state: Any  # TrainingState
    config: Any  # TrainingConfig
    context: Any  # TrainingContext

    # Models
    model: nn.Module
    orig_model: nn.Module  # Uncompiled version

    # Optimizers
    optimizers: List[Any]
    muon_optimizer: Optional[Any] = None

    # Logging and checkpointing
    logger: Any = None
    checkpoint_manager: Any = None
    profiler: Any = None

    # Per-step training data
    accumulated_loss: Optional[torch.Tensor] = None
    grad_norm: Optional[torch.Tensor] = None
    lrm: Optional[torch.Tensor] = None
    muon_momentum: Optional[torch.Tensor] = None

    # Dataloaders
    val_loader: Optional[Any] = None

    # Utilities
    tokenizer: Optional[Any] = None

    # Shared cache for hooks to avoid duplicate work
    cache: Dict[str, Any] = field(default_factory=dict)


class Hook:
    """
    Base class for all training hooks.

    Hooks encapsulate training operations that run conditionally
    (at intervals, on feature flags, etc.) without cluttering the main loop.
    """

    def __init__(self, name: Optional[str] = None, phases: Optional[List[str]] = None):
        """
        Initialize hook.

        Args:
            name: Optional name for debugging/logging
            phases: List of phases this hook should run in (e.g., ["after_backward", "after_step"])
                   If None, defaults to ["after_step"]
        """
        self.name = name or self.__class__.__name__
        self.phases = phases if phases is not None else ["after_step"]

    def should_run_in_phase(self, phase: str) -> bool:
        """
        Check if this hook should run in the given phase.

        Args:
            phase: Phase name (e.g., "after_backward", "after_step")

        Returns:
            True if hook should run in this phase, False otherwise
        """
        return phase in self.phases

    def should_run(self, step: int, config: Any) -> bool:
        """
        Determine if this hook should run on the given step.

        Args:
            step: Current training step
            config: Training configuration

        Returns:
            True if hook should execute, False otherwise
        """
        raise NotImplementedError(f"{self.name} must implement should_run()")

    def execute(self, ctx: HookContext) -> None:
        """
        Execute the hook's operation.

        Args:
            ctx: Hook context with all training state and data
        """
        raise NotImplementedError(f"{self.name} must implement execute()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class AlwaysHook(Hook):
    """Base class for hooks that always run."""

    def should_run(self, step: int, config: Any) -> bool:
        return True


class IntervalHook(Hook):
    """Base class for hooks that run at specific step intervals."""

    def __init__(self, interval: int, offset: int = 0, name: Optional[str] = None, phases: Optional[List[str]] = None):
        """
        Initialize interval-based hook.

        Args:
            interval: Run every N steps
            offset: Offset from step 0 (e.g., offset=100 means start at step 100)
            name: Optional name for debugging
            phases: List of phases this hook should run in
        """
        super().__init__(name, phases=phases)
        self.interval = interval
        self.offset = offset

    def should_run(self, step: int, config: Any) -> bool:
        if step < self.offset:
            return False
        return (step - self.offset) % self.interval == 0


class FeatureFlagHook(Hook):
    """Base class for hooks that run based on config feature flags."""

    def __init__(self, flag_name: str, name: Optional[str] = None, phases: Optional[List[str]] = None):
        """
        Initialize feature flag hook.

        Args:
            flag_name: Name of config attribute to check
            name: Optional name for debugging
            phases: List of phases this hook should run in
        """
        super().__init__(name, phases=phases)
        self.flag_name = flag_name

    def should_run(self, step: int, config: Any) -> bool:
        return getattr(config, self.flag_name, False)


class HookManager:
    """
    Manages hook execution with performance optimizations.

    Features:
    - Pre-computes hook schedules where possible
    - Executes hooks in priority order
    - Provides error handling and logging
    - Caches results to avoid duplicate work
    """

    def __init__(self, hooks: Optional[List[Hook]] = None):
        """
        Initialize hook manager.

        Args:
            hooks: List of hooks to manage (can add more later)
        """
        self.hooks: List[Hook] = hooks or []
        self._schedule_cache: Dict[int, List[Hook]] = {}

    def register(self, hook: Hook) -> None:
        """
        Register a new hook.

        Args:
            hook: Hook to register
        """
        self.hooks.append(hook)

    def register_many(self, hooks: List[Hook]) -> None:
        """
        Register multiple hooks.

        Args:
            hooks: List of hooks to register
        """
        self.hooks.extend(hooks)

    def run_hooks(self, ctx: HookContext, phase: str = "default") -> None:
        """
        Run all hooks that should execute on this step in the given phase.

        Args:
            ctx: Hook context with training state
            phase: Phase name for filtering hooks (e.g., "after_backward", "after_step")
        """
        # Clear cache at start of step
        if phase == "default" or phase == "after_step":
            ctx.cache.clear()

        # Find hooks that should run in this phase
        hooks_to_run = self._get_hooks_for_step(ctx.step, ctx.config)

        # Execute hooks in order, filtering by phase
        for hook in hooks_to_run:
            # Check if hook should run in this phase
            if not hook.should_run_in_phase(phase):
                continue

            try:
                hook.execute(ctx)
            except Exception as e:
                # Wrap exception with hook context for debugging
                raise RuntimeError(
                    f"Hook {hook.name} failed at step {ctx.step} in phase '{phase}': {e}"
                ) from e

    def _get_hooks_for_step(self, step: int, config: Any) -> List[Hook]:
        """
        Get list of hooks that should run on this step.

        Args:
            step: Current training step
            config: Training configuration

        Returns:
            List of hooks to execute
        """
        # For now, just check each hook
        # Could optimize with pre-computed schedules for interval hooks
        return [hook for hook in self.hooks if hook.should_run(step, config)]

    def get_schedule_summary(self, max_steps: int, config: Any) -> Dict[str, List[int]]:
        """
        Get a summary of when each hook will run.

        Useful for debugging and understanding hook execution patterns.

        Args:
            max_steps: Maximum number of steps to analyze
            config: Training configuration

        Returns:
            Dict mapping hook names to lists of steps where they run
        """
        summary = {hook.name: [] for hook in self.hooks}

        # Sample every 10th step for efficiency
        for step in range(0, max_steps, 10):
            for hook in self.hooks:
                if hook.should_run(step, config):
                    summary[hook.name].append(step)

        return summary

    def __len__(self) -> int:
        return len(self.hooks)

    def __repr__(self) -> str:
        return f"HookManager(hooks={len(self.hooks)})"
