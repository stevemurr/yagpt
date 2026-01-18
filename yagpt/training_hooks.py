"""
Training Hook Implementations

All hooks for managing training loop operations:
- Logging and metrics
- Evaluation and validation
- Checkpointing
- Sample generation
- Gradient management
- Debugging output
"""

import time
import math
import torch
from typing import Optional

from yagpt.hooks import Hook, HookContext, IntervalHook, FeatureFlagHook


class GradientClippingHook(Hook):
    """
    Gradient clipping and health checking.

    Clips gradients to prevent explosion and detects NaN/Inf gradients.
    Stores grad_norm in context for other hooks to use.

    Runs in: after_backward phase (needs gradients)
    """

    def __init__(self):
        super().__init__(name="GradientClippingHook", phases=["after_backward"])

    def should_run(self, step: int, config) -> bool:
        return config.grad_clip > 0.0

    def execute(self, ctx: HookContext) -> None:
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ctx.orig_model.parameters(),
            ctx.config.grad_clip
        )

        # Store in context for logging hooks
        ctx.grad_norm = grad_norm

        # Check for non-finite gradients
        if not torch.isfinite(grad_norm):
            print(f"\n{'='*70}")
            print(f"WARNING: Non-finite gradient norm detected at step {ctx.step}!")
            print(f"  grad_norm: {grad_norm.item()}")
            avg_loss = ctx.accumulated_loss.item() / ctx.context.grad_accum_steps
            print(f"  loss: {avg_loss}")
            print(f"{'='*70}\n")

            # Find which parameters have non-finite gradients
            for name, param in ctx.orig_model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    print(f"  Non-finite gradients in: {name}")


class LoggingHook(IntervalHook):
    """
    Main logging hook for training metrics.

    Handles:
    - Loss synchronization from GPU to CPU
    - EMA loss tracking
    - Timing metrics
    - Profiling data collection
    - Logging to all backends (wandb, tensorboard, csv)

    Runs in: after_step phase (after optimizer.step())
    """

    def __init__(self, interval: int = 10):
        super().__init__(interval=interval, name="LoggingHook", phases=["after_step"])

    def should_run(self, step: int, config) -> bool:
        # Use config's log_interval
        return step % config.log_interval == 0

    def execute(self, ctx: HookContext) -> None:
        # Average loss over gradient accumulation steps
        avg_loss = ctx.accumulated_loss / ctx.context.grad_accum_steps

        # Sync to CPU and update EMA
        ctx.state.update_train_loss(avg_loss.item(), ctx.config.ema_beta)

        # Calculate timing
        t1 = time.time()
        dt = t1 - ctx.state.last_log_time
        ctx.state.last_log_time = t1

        # Get profiling metrics
        perf_metrics = ctx.profiler.get_metrics()

        # Build log dictionary (enhanced with debugging info)
        log_dict = {
            'step': ctx.step,
            'train/loss': ctx.state.debiased_smooth_loss,
            'train/raw_loss': avg_loss.item(),  # Raw loss before EMA
            'train/smooth_loss': ctx.state.smooth_train_loss,  # EMA loss before debiasing
            'train/lrm': ctx.lrm.item(),
            'train/step_time_ms': dt * 1000,
            'perf/data_load_ms': perf_metrics.get('data_load_ms', 0),
            'perf/compute_ms': perf_metrics.get('compute_ms', 0),
            'perf/data_compute_ratio': perf_metrics.get('data_compute_ratio', 0),
        }

        # Add gradient norm if available
        if ctx.grad_norm is not None:
            log_dict['train/grad_norm'] = ctx.grad_norm.item() if torch.isfinite(ctx.grad_norm) else float('nan')

        # Add Muon momentum
        if ctx.muon_momentum is not None:
            log_dict['train/muon_momentum'] = ctx.muon_momentum.item()

        ctx.logger.log_metrics(log_dict, step=ctx.step)


class DebugLoggingHook(Hook):
    """
    Detailed debug output for first few training steps.

    Prints raw loss, smoothed loss, debiased loss, learning rate,
    momentum, gradients, and iteration counter for debugging.

    Runs in: after_step phase (after optimizer.step())
    """

    def __init__(self, num_steps: int = 5):
        super().__init__(name="DebugLoggingHook", phases=["after_step"])
        self.num_steps = num_steps

    def should_run(self, step: int, config) -> bool:
        # Run for first N steps, but only on log intervals
        return step < self.num_steps and step % config.log_interval == 0

    def execute(self, ctx: HookContext) -> None:
        avg_loss = ctx.accumulated_loss / ctx.context.grad_accum_steps

        print(f"\n{'='*70}")
        print(f"DEBUG Step {ctx.step}:")
        print(f"  Raw Loss: {avg_loss.item():.6f}")
        print(f"  Smooth Loss: {ctx.state.smooth_train_loss:.6f}")
        print(f"  Debiased Loss: {ctx.state.debiased_smooth_loss:.6f}")
        print(f"  LR Multiplier: {ctx.lrm.item():.6f}")

        if ctx.muon_momentum is not None:
            print(f"  Muon Momentum: {ctx.muon_momentum.item():.6f}")

        if ctx.grad_norm is not None:
            print(f"  Grad Norm: {ctx.grad_norm.item():.6f}")

        print(f"  Iteration Counter: {ctx.state.iteration}")
        print(f"{'='*70}\n")


class ProfilingAnalysisHook(Hook):
    """
    Detailed profiling analysis at regular intervals.

    Prints profiling breakdown every N steps to help identify
    bottlenecks in data loading vs compute.

    Runs in: after_step phase (after optimizer.step())
    """

    def __init__(self):
        super().__init__(name="ProfilingAnalysisHook", phases=["after_step"])

    def should_run(self, step: int, config) -> bool:
        return step % (config.log_interval * 10) == 0 and step > 0

    def execute(self, ctx: HookContext) -> None:
        perf_metrics = ctx.profiler.get_metrics()
        ctx.profiler.print_analysis(ctx.step, ctx.context.num_workers, perf_metrics)


class EvaluationHook(IntervalHook):
    """
    Validation evaluation hook.

    Runs validation loop, calculates perplexity, updates best loss tracking,
    and logs validation metrics. Caches validation results for other hooks.

    Runs in: after_step phase (eval after step completes)
    """

    def __init__(self, interval: int = 5000):
        super().__init__(interval=interval, offset=1, name="EvaluationHook", phases=["after_step"])

    def should_run(self, step: int, config) -> bool:
        # step > 0 and step % eval_interval == 0
        return step > 0 and step % config.eval_interval == 0

    def execute(self, ctx: HookContext) -> None:
        # Import here to avoid circular dependency
        from yagpt.train import estimate_loss

        # Check cache to avoid duplicate evaluation
        if 'val_loss' in ctx.cache:
            return

        # Run validation
        val_loss = estimate_loss(
            ctx.model,
            ctx.val_loader,
            ctx.config.eval_iters,
            ctx.context
        )

        val_perplexity = math.exp(val_loss)

        # Update validation state
        ctx.state.update_val_loss(val_loss)

        # Cache for other hooks (e.g., CheckpointHook)
        ctx.cache['val_loss'] = val_loss

        # Log validation metrics
        ctx.logger.log_metrics({
            'val/loss': val_loss,
            'val/perplexity': val_perplexity,
        }, step=ctx.step)

        if val_loss < ctx.state.best_val_loss:
            ctx.logger.log_metrics({
                'val/best_loss': ctx.state.best_val_loss,
            }, step=ctx.step)


class SampleGenerationHook(Hook):
    """
    Text sample generation hook.

    Generates sample texts from the model for qualitative monitoring.
    Only runs when generate_samples config flag is enabled.

    Runs in: after_step phase (generate after step completes)
    """

    def __init__(self):
        super().__init__(name="SampleGenerationHook", phases=["after_step"])

    def should_run(self, step: int, config) -> bool:
        # Only run during evaluation intervals and if feature is enabled
        return (config.generate_samples and
                step > 0 and
                step % config.eval_interval == 0)

    def execute(self, ctx: HookContext) -> None:
        # Import here to avoid circular dependency
        from yagpt.train import generate_sample_texts

        sample_text = generate_sample_texts(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            config=ctx.config,
            context=ctx.context,
            iteration=ctx.step
        )

        ctx.logger.log_text(name="samples/generated_text", text=sample_text, step=ctx.step)
        # Also print to console for immediate feedback
        print(sample_text)


class CheckpointHook(IntervalHook):
    """
    Checkpoint saving hook.

    Saves model, optimizers, and training state at regular intervals.
    Ensures fresh validation loss is available before checkpointing.

    Runs in: after_step phase (checkpoint after step completes)
    """

    def __init__(self, interval: int = 1000):
        super().__init__(interval=interval, offset=1, name="CheckpointHook", phases=["after_step"])

    def should_run(self, step: int, config) -> bool:
        return step > 0 and step % config.checkpoint_interval == 0

    def execute(self, ctx: HookContext) -> None:
        # Import here to avoid circular dependency
        from yagpt.train import estimate_loss

        # Ensure we have recent validation loss
        # Check if we need to run evaluation (not on eval interval or no cached result)
        needs_eval = (
            ctx.state.last_val_loss is None or
            (ctx.step % ctx.config.eval_interval) != 0
        )

        # Check cache first (EvaluationHook may have run)
        if needs_eval and 'val_loss' not in ctx.cache:
            val_loss = estimate_loss(
                ctx.model,
                ctx.val_loader,
                ctx.config.eval_iters,
                ctx.context
            )
            ctx.state.update_val_loss(val_loss)
            ctx.cache['val_loss'] = val_loss

        # Save checkpoint
        ctx.checkpoint_manager.save_checkpoint(
            model=ctx.orig_model,
            optimizers=ctx.optimizers,
            state=ctx.state,
            config=ctx.config,
        )
        print()


class GradientMonitoringHook(Hook):
    """
    Monitor gradient magnitudes for different parameter groups.

    Helps diagnose gradient flow issues and optimizer balance.
    Logs gradient norms for embeddings vs transformer blocks.

    Runs in: after_backward phase (needs gradients)
    """

    def __init__(self, log_interval: int = 100):
        super().__init__(name="GradientMonitoringHook", phases=["after_backward"])
        self.log_interval = log_interval

    def should_run(self, step: int, config) -> bool:
        return step % self.log_interval == 0

    def execute(self, ctx: HookContext) -> None:
        # Calculate gradient norms for different parameter groups
        embedding_grad_norm = 0.0
        transformer_grad_norm = 0.0
        lm_head_grad_norm = 0.0

        # Embeddings
        for p in ctx.orig_model.transformer.wte.parameters():
            if p.grad is not None:
                embedding_grad_norm += p.grad.norm().item() ** 2
        embedding_grad_norm = embedding_grad_norm ** 0.5

        # Transformer blocks
        for p in ctx.orig_model.transformer.h.parameters():
            if p.grad is not None:
                transformer_grad_norm += p.grad.norm().item() ** 2
        transformer_grad_norm = transformer_grad_norm ** 0.5

        # LM head (if not weight-tied)
        lm_head_param_ids = {id(p) for p in ctx.orig_model.transformer.wte.parameters()}
        for p in ctx.orig_model.lm_head.parameters():
            if p.grad is not None and id(p) not in lm_head_param_ids:
                lm_head_grad_norm += p.grad.norm().item() ** 2
        lm_head_grad_norm = lm_head_grad_norm ** 0.5

        # Log gradient norms
        ctx.logger.log_metrics({
            'gradients/embedding_norm': embedding_grad_norm,
            'gradients/transformer_norm': transformer_grad_norm,
            'gradients/lm_head_norm': lm_head_grad_norm,
            'gradients/emb_to_transformer_ratio': embedding_grad_norm / (transformer_grad_norm + 1e-8),
        }, step=ctx.step)

        # Print warning if gradients are severely imbalanced
        if ctx.step < 1000:  # Only check in early training
            ratio = embedding_grad_norm / (transformer_grad_norm + 1e-8)
            if ratio < 0.01 or ratio > 100:
                print(f"\n⚠️  WARNING: Gradient imbalance detected at step {ctx.step}")
                print(f"   Embedding grad norm: {embedding_grad_norm:.6f}")
                print(f"   Transformer grad norm: {transformer_grad_norm:.6f}")
                print(f"   Ratio: {ratio:.6f}")
                print(f"   This may indicate optimizer or learning rate issues.\n")


# Hook factory for easy configuration
def create_default_hooks(config) -> list:
    """
    Create the default set of training hooks.

    Args:
        config: Training configuration

    Returns:
        List of hooks configured for standard training
    """
    hooks = [
        # Core training hooks (always needed)
        GradientClippingHook(),
        LoggingHook(interval=config.log_interval),

        # Debugging hooks
        DebugLoggingHook(num_steps=5),
        ProfilingAnalysisHook(),
        GradientMonitoringHook(log_interval=100),

        # Evaluation and monitoring
        EvaluationHook(interval=config.eval_interval),
        SampleGenerationHook(),

        # Checkpointing
        CheckpointHook(interval=config.checkpoint_interval),
    ]

    return hooks
