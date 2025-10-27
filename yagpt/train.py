"""
Training script for YAGPT

Includes:
- Training loop with gradient accumulation
- Learning rate scheduling (warmup + cosine decay)
- Checkpointing (keeps last 5 checkpoints)
- Resume training from checkpoint
- Evaluation loop
- Logging and metrics
"""

import os
import time
import math
import glob
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yagpt.model import GPT, GPTConfig, create_gpt_mini
from yagpt.tokenizer import GPT4Tokenizer
from yagpt.dataloader import FineWebDataset
from yagpt.logger import create_logger

# Enable TF32 for faster training on Ampere+ GPUs (RTX 3000/4000 series)
# TF32 uses lower precision for matmul but maintains full precision for accumulation
# Provides ~3x speedup with minimal accuracy impact
torch.set_float32_matmul_precision('high')  # 'highest', 'high', or 'medium'

# Alternative explicit settings (choose one approach):
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "gpt_mini"
    vocab_size: int = 100277
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 2048
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch = 8 * 4 = 32
    max_iters: int = 100000

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 3e-5

    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 100

    # Text generation during evaluation
    generate_samples: bool = True  # Generate text samples during eval
    num_generation_prompts: int = 3  # Number of prompts to generate from
    generation_max_tokens: int = 100  # Max tokens per generation
    generation_temperature: float = 0.8
    generation_top_k: int = 40
    generation_top_p: float = 0.9

    # Logging
    log_interval: int = 10
    log_backends: list[str] = None  # Will be set in __post_init__
    wandb_project: str = "gpt-mini"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", "disabled"
    tensorboard_dir: str = "./runs"
    csv_log_dir: str = "./logs"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000
    keep_last_n_checkpoints: int = 5

    # Data
    data_dir: str = "./datasets/fineweb"
    train_data_dir: Optional[str] = None  # If None, uses data_dir
    val_data_dir: Optional[str] = None    # If None, uses data_dir (not recommended)
    num_workers: int = 4

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)

    # Resume
    resume_from: Optional[str] = None

    def __post_init__(self):
        # Set default log backends if not specified
        if self.log_backends is None:
            self.log_backends = ['console', 'wandb']

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'TrainingConfig':
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            TrainingConfig instance

        Example:
            config = TrainingConfig.from_yaml("configs/optimized-128gb.yaml")
            train(config)
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Flatten nested structure into TrainingConfig parameters
        params = {}

        # Model parameters
        if 'model' in yaml_config:
            model = yaml_config['model']
            params['model_name'] = model.get('name', 'gpt_mini')
            params['vocab_size'] = model.get('vocab_size', 100277)
            params['n_layer'] = model.get('n_layer', 12)
            params['n_head'] = model.get('n_head', 12)
            params['n_embd'] = model.get('n_embd', 768)
            params['block_size'] = model.get('block_size', 2048)
            params['dropout'] = model.get('dropout', 0.1)

        # Training parameters
        if 'training' in yaml_config:
            training = yaml_config['training']
            params['batch_size'] = training.get('batch_size', 8)
            params['gradient_accumulation_steps'] = training.get('gradient_accumulation_steps', 4)
            params['max_iters'] = training.get('max_iters', 100000)
            params['learning_rate'] = training.get('learning_rate', 3e-4)
            params['weight_decay'] = training.get('weight_decay', 0.1)
            params['beta1'] = training.get('beta1', 0.9)
            params['beta2'] = training.get('beta2', 0.95)
            params['grad_clip'] = training.get('grad_clip', 1.0)
            params['warmup_iters'] = training.get('warmup_iters', 2000)
            params['lr_decay_iters'] = training.get('lr_decay_iters', 100000)
            params['min_lr'] = training.get('min_lr', 3e-5)

        # Evaluation parameters
        if 'evaluation' in yaml_config:
            evaluation = yaml_config['evaluation']
            params['eval_interval'] = evaluation.get('eval_interval', 500)
            params['eval_iters'] = evaluation.get('eval_iters', 100)
            params['generate_samples'] = evaluation.get('generate_samples', True)
            params['num_generation_prompts'] = evaluation.get('num_generation_prompts', 3)
            params['generation_max_tokens'] = evaluation.get('generation_max_tokens', 100)
            params['generation_temperature'] = evaluation.get('generation_temperature', 0.8)
            params['generation_top_k'] = evaluation.get('generation_top_k', 40)
            params['generation_top_p'] = evaluation.get('generation_top_p', 0.9)

        # Logging parameters
        if 'logging' in yaml_config:
            logging = yaml_config['logging']
            params['log_interval'] = logging.get('log_interval', 10)
            params['log_backends'] = logging.get('backends', ['console', 'wandb'])
            params['wandb_project'] = logging.get('wandb_project', 'gpt-mini')
            params['wandb_entity'] = logging.get('wandb_entity')
            params['wandb_mode'] = logging.get('wandb_mode', 'online')
            params['tensorboard_dir'] = logging.get('tensorboard_dir', './runs')
            params['csv_log_dir'] = logging.get('csv_log_dir', './logs')

        # Checkpointing parameters
        if 'checkpointing' in yaml_config:
            checkpointing = yaml_config['checkpointing']
            params['checkpoint_dir'] = checkpointing.get('checkpoint_dir', './checkpoints')
            params['checkpoint_interval'] = checkpointing.get('checkpoint_interval', 1000)
            params['keep_last_n_checkpoints'] = checkpointing.get('keep_last_n_checkpoints', 5)

        # Data parameters
        if 'data' in yaml_config:
            data = yaml_config['data']
            params['data_dir'] = data.get('data_dir', './datasets/fineweb')
            params['train_data_dir'] = data.get('train_data_dir')
            params['val_data_dir'] = data.get('val_data_dir')
            params['num_workers'] = data.get('num_workers', 4)

        # System parameters
        if 'system' in yaml_config:
            system = yaml_config['system']
            params['device'] = system.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            params['compile_model'] = system.get('compile', False)

        return cls(**params)


class CheckpointManager:
    """Manages model checkpoints, keeping only the last N."""

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        config: TrainingConfig,
        train_loss: float,
        val_loss: Optional[float] = None,
    ) -> str:
        """Save a checkpoint and manage old checkpoints."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pt"

        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': asdict(config),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_iter_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        # Remove old checkpoints
        for ckpt in checkpoints[:-self.keep_last_n]:
            ckpt.unlink()
            print(f"Removed old checkpoint: {ckpt}")

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the most recent checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_iter_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        if checkpoints:
            return str(checkpoints[-1])
        return None

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        iteration = checkpoint['iteration']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint.get('val_loss')

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Iteration: {iteration}")
        print(f"  Train loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Val loss: {val_loss:.4f}")

        return iteration, train_loss, val_loss


def get_lr(it: int, config: TrainingConfig) -> float:
    """
    Learning rate schedule: warmup + cosine decay.

    1) Linear warmup for warmup_iters steps
    2) Cosine decay to min_lr for remaining steps
    """
    # 1) Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters

    # 2) After lr_decay_iters, return min_lr
    if it > config.lr_decay_iters:
        return config.min_lr

    # 3) In between, use cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model: nn.Module, dataloader: DataLoader, eval_iters: int, device: str) -> float:
    """Estimate average loss over eval_iters batches."""
    model.eval()
    losses = []

    for i, batch in enumerate(dataloader):
        if i >= eval_iters:
            break

        # Get input and target (shift by 1 for language modeling)
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)

        # Forward pass (KV cache not used during evaluation)
        _, loss, _ = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else 0.0


@torch.no_grad()
def generate_sample_texts(
    model: nn.Module,
    tokenizer,
    config: TrainingConfig,
    device: str,
    iteration: int
) -> str:
    """
    Generate sample texts from the model for monitoring during training.

    Args:
        model: The GPT model
        tokenizer: Tokenizer for encoding/decoding
        config: Training configuration
        device: Device to run on
        iteration: Current training iteration

    Returns:
        Formatted string with all generated samples
    """
    model.eval()

    # Sample prompts for generation
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time,",
        "In the year 2050,",
        "The key to happiness is",
        "Science has shown that",
        "The most important thing in life is",
    ]

    # Use only the configured number of prompts
    prompts = prompts[:config.num_generation_prompts]

    output_lines = []
    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"Generated Samples at Iteration {iteration}")
    output_lines.append(f"{'='*70}\n")

    for i, prompt in enumerate(prompts, 1):
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        # Generate (use_cache=False during training to avoid dropout issues)
        # KV cache can have issues with dropout layers during eval mode
        generated_ids = model.generate(
            idx,
            max_new_tokens=config.generation_max_tokens,
            temperature=config.generation_temperature,
            top_k=config.generation_top_k,
            use_cache=False  # Disable cache during training generation
        )

        # Decode
        generated_text = tokenizer.decode(generated_ids[0].tolist())

        # Format output
        output_lines.append(f"[{i}/{len(prompts)}] Prompt: {prompt}")
        output_lines.append(f"Generated: {generated_text}")
        output_lines.append(f"{'-'*70}\n")

    model.train()
    return "\n".join(output_lines)


def train(config: TrainingConfig):
    """Main training loop."""
    print("="*70)
    print("GPT-4 Mini Training")
    print("="*70)
    print(f"\nConfiguration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print()

    # Set device
    device = config.device
    print(f"Using device: {device}")

    # Create tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = GPT4Tokenizer()

    # Create datasets
    print("Creating datasets...")

    # Determine train and val data directories
    train_dir = config.train_data_dir if config.train_data_dir else config.data_dir
    val_dir = config.val_data_dir if config.val_data_dir else config.data_dir

    # Warn if using same directory for train and val
    if train_dir == val_dir:
        print(f"WARNING: Using the same data directory for both training and validation: {train_dir}")
        print("This means validation loss will NOT be a true measure of generalization!")
        print("Consider setting train_data_dir and val_data_dir to separate directories.")

    print(f"Train data directory: {train_dir}")
    print(f"Validation data directory: {val_dir}")

    train_dataset = FineWebDataset(
        data_dir=train_dir,
        seq_length=config.block_size,
        tokenizer=tokenizer,
        shuffle_shards=True,
    )

    val_dataset = FineWebDataset(
        data_dir=val_dir,
        seq_length=config.block_size,
        tokenizer=tokenizer,
        shuffle_shards=False,  # Don't shuffle val data for reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Create model
    print("\nInitializing model...")
    model_config = GPTConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        dropout=config.dropout,
    )
    model = GPT(model_config)
    model = model.to(device)

    # Compile model (NOTE: on dgx spark must use pytorch nightlys for now 10.27.25)
    model = torch.compile(model, backend="cudagraphs")

    # Create optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device,
    )

    # Create logger
    print("\nInitializing logger...")
    logger = create_logger(
        backends=config.log_backends,
        log_dir=config.csv_log_dir,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_mode=config.wandb_mode,
        config=asdict(config),
    )

    # Log initial configuration
    logger.log_config(asdict(config))

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        config.checkpoint_dir,
        keep_last_n=config.keep_last_n_checkpoints,
    )

    # Resume from checkpoint if specified
    start_iter = 0
    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        start_iter, _, _ = checkpoint_manager.load_checkpoint(
            config.resume_from, model, optimizer
        )
        start_iter += 1  # Start from next iteration

    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    model.train()
    train_iter = iter(train_loader)

    best_val_loss = float('inf')
    t0 = time.time()

    for iteration in range(start_iter, config.max_iters):
        # Mark step boundary for CUDA graphs (required with torch.compile)
        torch.compiler.cudagraph_mark_step_begin()

        # Update learning rate
        lr = get_lr(iteration, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation loop
        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(config.gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Prepare input and target
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)

            # Forward pass (KV cache not used during training)
            _, loss, _ = model(x, y)

            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss_accum += loss.item()

            # Backward pass
            loss.backward()

        # Clip gradients
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update parameters
        optimizer.step()

        # Logging
        if iteration % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Log metrics
            logger.log_metrics({
                'train/loss': loss_accum,
                'train/lr': lr,
                'train/step_time_ms': dt * 1000,
            }, step=iteration)

        # Evaluation
        if iteration > 0 and iteration % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, config.eval_iters, device)

            # Log validation metrics
            logger.log_metrics({
                'val/loss': val_loss,
            }, step=iteration)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.log_metrics({
                    'val/best_loss': best_val_loss,
                }, step=iteration)

            # Generate sample texts for monitoring
            if config.generate_samples:
                sample_text = generate_sample_texts(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    device=device,
                    iteration=iteration
                )
                # Log to W&B and other backends
                logger.log_text(name="samples/generated_text", text=sample_text, step=iteration)
                # Also print to console for immediate feedback
                print(sample_text)

        # Checkpointing
        if iteration > 0 and iteration % config.checkpoint_interval == 0:
            val_loss = estimate_loss(model, val_loader, config.eval_iters, device)
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                config=config,
                train_loss=loss_accum,
                val_loss=val_loss,
            )
            print()

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    val_loss = estimate_loss(model, val_loader, config.eval_iters, device)
    checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=config.max_iters,
        config=config,
        train_loss=loss_accum,
        val_loss=val_loss,
    )

    # Finish logging
    logger.finish()


def main():
    """Entry point."""
    # Create default config
    config = TrainingConfig()

    # You can override config here or via command line arguments
    # For example:
    # config.batch_size = 16
    # config.resume_from = "./checkpoints/checkpoint_iter_5000.pt"

    # Start training
    train(config)


if __name__ == "__main__":
    main()
