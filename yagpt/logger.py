"""
Logging system for GPT-4 Mini training

Supports multiple backends:
- Console: Real-time terminal output
- CSV: Simple metrics archival
- W&B: Weights & Biases (optional, requires signup)
- TensorBoard: Local visualization (optional)

All loggers share a common interface and can be used simultaneously.
"""

import os
import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from dataclasses import asdict


class BaseLogger(ABC):
    """
    Abstract base class for all loggers.

    All concrete loggers must implement these methods.
    """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log scalar metrics at a given step."""
        pass

    @abstractmethod
    def log_config(self, config: dict[str, Any]):
        """Log configuration/hyperparameters once at start."""
        pass

    @abstractmethod
    def log_text(self, name: str, text: str, step: int):
        """Log text samples (e.g., generated text)."""
        pass

    @abstractmethod
    def finish(self):
        """Clean up and finalize logging."""
        pass


class ConsoleLogger(BaseLogger):
    """
    Simple console logger that prints to terminal.

    Features:
    - Color-coded output (if terminal supports it)
    - Formatted metrics display
    - Progress indication
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = datetime.now()

        # ANSI color codes
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'cyan': '\033[96m',
        }

        # Check if terminal supports colors
        self.use_colors = os.getenv('TERM') and os.isatty(1)

        print(self._colorize("="*70, 'blue'))
        print(self._colorize("Console Logger Initialized", 'bold'))
        print(self._colorize("="*70, 'blue'))
        print()

    def _colorize(self, text: str, color: str) -> str:
        """Add color to text if terminal supports it."""
        if self.use_colors and color in self.colors:
            return f"{self.colors[color]}{text}{self.colors['reset']}"
        return text

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to console."""
        if not self.verbose:
            return

        # Format metrics nicely
        metric_strs = []
        for key, value in metrics.items():
            # Short key name (remove prefix if present)
            short_key = key.split('/')[-1]

            # Format value based on magnitude
            if isinstance(value, float):
                if abs(value) < 0.01 or abs(value) > 1000:
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            metric_strs.append(f"{short_key}={value_str}")

        # Create log line
        step_str = self._colorize(f"step {step:6d}", 'cyan')
        metrics_str = " | ".join(metric_strs)

        print(f"{step_str} | {metrics_str}")

    def log_config(self, config: dict[str, Any]):
        """Log configuration to console."""
        print(self._colorize("\nConfiguration:", 'bold'))
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

    def log_text(self, name: str, text: str, step: int):
        """Log text sample to console."""
        print(f"\n{self._colorize(f'[Step {step}] {name}:', 'yellow')}")
        print(text[:500] + ("..." if len(text) > 500 else ""))
        print()

    def finish(self):
        """Print summary and finish."""
        elapsed = datetime.now() - self.start_time
        print(self._colorize("="*70, 'blue'))
        print(self._colorize(f"Logging finished. Elapsed time: {elapsed}", 'green'))
        print(self._colorize("="*70, 'blue'))


class CSVLogger(BaseLogger):
    """
    CSV logger for simple metrics archival.

    Creates one CSV file with all metrics, where each row is a step.
    Easy to load into pandas/Excel for analysis.
    """

    def __init__(self, log_dir: str = "./logs", run_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create unique run name
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        self.run_name = run_name
        self.metrics_file = self.log_dir / f"{run_name}_metrics.csv"
        self.config_file = self.log_dir / f"{run_name}_config.json"

        # CSV state
        self.csv_file = None
        self.csv_writer = None
        self.fieldnames = ['step']  # Will be extended as we see new metrics

        print(f"CSV Logger: Logging to {self.metrics_file}")

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to CSV file."""
        # Add step to metrics
        row = {'step': step, **metrics}

        # Check if we need to add new columns
        new_fields = [k for k in row.keys() if k not in self.fieldnames]
        if new_fields:
            self.fieldnames.extend(new_fields)
            # Need to rewrite file with new headers if file exists
            if self.csv_file:
                self._rewrite_with_new_fields()

        # Open file if not already open
        if self.csv_file is None:
            self.csv_file = open(self.metrics_file, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()

        # Write row
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure written to disk

    def _rewrite_with_new_fields(self):
        """Rewrite CSV with new field names."""
        # Read existing data
        self.csv_file.close()
        existing_data = []
        with open(self.metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)

        # Rewrite with new fieldnames
        self.csv_file = open(self.metrics_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.csv_writer.writeheader()
        self.csv_writer.writerows(existing_data)

    def log_config(self, config: dict[str, Any]):
        """Save configuration to JSON file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"CSV Logger: Config saved to {self.config_file}")

    def log_text(self, name: str, text: str, step: int):
        """Log text to a separate file."""
        text_file = self.log_dir / f"{self.run_name}_samples.txt"
        with open(text_file, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Step {step} - {name}\n")
            f.write(f"{'='*70}\n")
            f.write(text)
            f.write("\n\n")

    def finish(self):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
        print(f"CSV Logger: Metrics saved to {self.metrics_file}")


class WandBLogger(BaseLogger):
    """
    Weights & Biases logger (optional).

    Requires:
    - pip install wandb
    - wandb login (or WANDB_API_KEY env var)

    Falls back gracefully if wandb is not available.
    """

    def __init__(
        self,
        project: str = "gpt-mini",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        mode: str = "online",  # "online", "offline", "disabled"
        config: Optional[dict] = None,
    ):
        self.enabled = False
        self.run = None

        try:
            import wandb
            self.wandb = wandb

            # Initialize run
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                mode=mode,
                config=config,
                resume="allow",  # Allow resuming runs
            )

            self.enabled = True
            print(f"W&B Logger: Initialized (mode={mode}, project={project})")
            if mode == "online":
                print(f"W&B Logger: View at {self.run.get_url()}")

        except ImportError:
            print("W&B Logger: wandb package not found. Install with: pip install wandb")
            print("W&B Logger: Falling back to other loggers")

        except Exception as e:
            print(f"W&B Logger: Failed to initialize: {e}")
            print("W&B Logger: Run 'wandb login' if not authenticated")
            print("W&B Logger: Falling back to other loggers")

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to W&B."""
        if not self.enabled:
            return

        try:
            self.wandb.log(metrics, step=step)
        except Exception as e:
            print(f"W&B Logger: Error logging metrics: {e}")

    def log_config(self, config: dict[str, Any]):
        """Update W&B config (already set in init, but can update)."""
        if not self.enabled:
            return

        try:
            self.wandb.config.update(config)
        except Exception as e:
            print(f"W&B Logger: Error logging config: {e}")

    def log_text(self, name: str, text: str, step: int):
        """Log text as W&B artifact."""
        if not self.enabled:
            return

        try:
            self.wandb.log({name: self.wandb.Html(f"<pre>{text}</pre>")}, step=step)
        except Exception as e:
            print(f"W&B Logger: Error logging text: {e}")

    def finish(self):
        """Finish W&B run."""
        if self.enabled and self.run:
            self.wandb.finish()
            print("W&B Logger: Run finished")


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard logger (optional).

    Requires:
    - pip install tensorboard

    View with: tensorboard --logdir=./runs
    """

    def __init__(self, log_dir: str = "./runs", run_name: Optional[str] = None):
        self.enabled = False
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            # Create unique run name
            if run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"run_{timestamp}"

            log_path = Path(log_dir) / run_name
            self.writer = SummaryWriter(log_dir=str(log_path))

            self.enabled = True
            print(f"TensorBoard Logger: Logging to {log_path}")
            print(f"TensorBoard Logger: View with 'tensorboard --logdir={log_dir}'")

        except ImportError:
            print("TensorBoard Logger: tensorboard not found. Install with: pip install tensorboard")
            print("TensorBoard Logger: Falling back to other loggers")

        except Exception as e:
            print(f"TensorBoard Logger: Failed to initialize: {e}")
            print("TensorBoard Logger: Falling back to other loggers")

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to TensorBoard."""
        if not self.enabled:
            return

        try:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        except Exception as e:
            print(f"TensorBoard Logger: Error logging metrics: {e}")

    def log_config(self, config: dict[str, Any]):
        """Log config as text to TensorBoard."""
        if not self.enabled:
            return

        try:
            config_str = json.dumps(config, indent=2, default=str)
            self.writer.add_text("config", config_str)
        except Exception as e:
            print(f"TensorBoard Logger: Error logging config: {e}")

    def log_text(self, name: str, text: str, step: int):
        """Log text to TensorBoard."""
        if not self.enabled:
            return

        try:
            self.writer.add_text(name, text, step)
        except Exception as e:
            print(f"TensorBoard Logger: Error logging text: {e}")

    def finish(self):
        """Close TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()
            print("TensorBoard Logger: Closed")


class MultiLogger:
    """
    Composite logger that wraps multiple backends.

    Forwards all logging calls to all registered backends.
    """

    def __init__(self, loggers: list[BaseLogger]):
        self.loggers = loggers
        print(f"\nMultiLogger: Initialized with {len(loggers)} backend(s)")
        print()

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log metrics to all backends."""
        for logger in self.loggers:
            try:
                logger.log_metrics(metrics, step)
            except Exception as e:
                print(f"Error in {logger.__class__.__name__}: {e}")

    def log_config(self, config: dict[str, Any]):
        """Log config to all backends."""
        for logger in self.loggers:
            try:
                logger.log_config(config)
            except Exception as e:
                print(f"Error in {logger.__class__.__name__}: {e}")

    def log_text(self, name: str, text: str, step: int):
        """Log text to all backends."""
        for logger in self.loggers:
            try:
                logger.log_text(name, text, step)
            except Exception as e:
                print(f"Error in {logger.__class__.__name__}: {e}")

    def finish(self):
        """Finish all backends."""
        for logger in self.loggers:
            try:
                logger.finish()
            except Exception as e:
                print(f"Error in {logger.__class__.__name__}: {e}")


def create_logger(
    backends: list[str],
    run_name: Optional[str] = None,
    log_dir: str = "./logs",
    wandb_project: str = "gpt-mini",
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
    config: Optional[dict] = None,
) -> MultiLogger:
    """
    Factory function to create a logger with specified backends.

    Args:
        backends: List of backend names: 'console', 'csv', 'wandb', 'tensorboard'
        run_name: Optional run name (auto-generated if None)
        log_dir: Directory for CSV and TensorBoard logs
        wandb_project: W&B project name
        wandb_entity: W&B entity (team/username)
        wandb_mode: W&B mode ('online', 'offline', 'disabled')
        config: Initial configuration to log

    Returns:
        MultiLogger instance with requested backends

    Example:
        logger = create_logger(
            backends=['console', 'csv', 'wandb'],
            wandb_project='my-gpt-project',
            config=config_dict,
        )
    """
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    loggers = []

    # Always create console logger first for immediate feedback
    if 'console' in backends:
        loggers.append(ConsoleLogger())

    # CSV logger (no dependencies)
    if 'csv' in backends:
        loggers.append(CSVLogger(log_dir=log_dir, run_name=run_name))

    # W&B logger (optional, may fail gracefully)
    if 'wandb' in backends:
        loggers.append(WandBLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            mode=wandb_mode,
            config=config,
        ))

    # TensorBoard logger (optional, may fail gracefully)
    if 'tensorboard' in backends:
        loggers.append(TensorBoardLogger(log_dir=log_dir, run_name=run_name))

    if not loggers:
        # Fallback to console if no backends specified
        print("Warning: No logging backends specified, using console")
        loggers.append(ConsoleLogger())

    return MultiLogger(loggers)


if __name__ == "__main__":
    # Test the logging system
    print("Testing Logging System\n")

    # Create logger with all backends
    logger = create_logger(
        backends=['console', 'csv', 'wandb', 'tensorboard'],
        wandb_project='gpt-mini-test',
        wandb_mode='offline',  # Use offline mode for testing
    )

    # Log config
    test_config = {
        'model': 'gpt-mini',
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'learning_rate': 3e-4,
    }
    logger.log_config(test_config)

    # Simulate training loop
    print("\nSimulating training loop...\n")
    for step in range(10):
        metrics = {
            'train/loss': 4.5 - step * 0.1,
            'train/lr': 3e-4 * (1 - step/10),
            'train/step_time_ms': 250 + step * 5,
        }
        logger.log_metrics(metrics, step)

        # Log validation metrics every 5 steps
        if step % 5 == 0:
            val_metrics = {
                'val/loss': 4.6 - step * 0.1,
                'val/perplexity': 100 - step * 5,
            }
            logger.log_metrics(val_metrics, step)

    # Log text sample
    logger.log_text(
        name="generated_sample",
        text="The quick brown fox jumps over the lazy dog. This is a test of text logging.",
        step=10,
    )

    # Finish
    logger.finish()

    print("\n✅ Logging test complete!")
    print("\nCheck outputs:")
    print("  - Console: Printed above")
    print("  - CSV: ./logs/run_*_metrics.csv")
    print("  - W&B: ./wandb/ (offline mode)")
    print("  - TensorBoard: ./runs/run_*")
