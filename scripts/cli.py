#!/usr/bin/env python3
"""
YAGPT Command Line Interface

Train, fine-tune, and generate text with YAGPT models.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import typer
import yaml
from rich.console import Console
from rich.table import Table

from yagpt.train import TrainingConfig, train as run_training
from yagpt.generate import load_model_from_checkpoint, generate_text
from yagpt.tokenizer import GPT4Tokenizer
from yagpt.model import create_gpt_mini

app = typer.Typer(
    name="yagpt",
    help="YAGPT - Yet Another GPT implementation for education and experimentation",
    add_completion=False,
)
console = Console()


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint in a directory."""
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_iter_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    return checkpoints[-1] if checkpoints else None


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file",
        exists=True,
        dir_okay=False,
    ),
):
    """
    Train a model from scratch with default or custom configuration.

    Examples:
        yagpt train
        yagpt train --config configs/optimized-128gb.yaml
    """
    if config:
        console.print(f"[bold green]Loading configuration from:[/bold green] {config}")
        train_config = TrainingConfig.from_yaml(str(config))

        console.print(f"\n[bold]Training Configuration:[/bold]")
        console.print(f"  Model: {train_config.model_name}")
        console.print(f"  Batch size: {train_config.batch_size}")
        console.print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
        console.print(f"  Effective batch: {train_config.batch_size * train_config.gradient_accumulation_steps}")
        console.print(f"  Max iterations: {train_config.max_iters}\n")
    else:
        console.print("[bold green]Training with default configuration[/bold green]\n")
        train_config = TrainingConfig(
            # Model: Mini configuration
            model_name="gpt_mini",
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=2048,
            # Training
            batch_size=8,
            gradient_accumulation_steps=4,
            max_iters=100000,
            # Data
            data_dir="./datasets/fineweb",
            # Checkpointing
            checkpoint_interval=1000,
            keep_last_n_checkpoints=5,
            # Evaluation
            eval_interval=500,
        )

    run_training(train_config)


@app.command()
def resume(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file (uses config's checkpoint_dir)",
        exists=True,
        dir_okay=False,
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        "--resume-from-checkpoint",
        help="Specific checkpoint to resume from (overrides auto-detection)",
        exists=True,
        dir_okay=False,
    ),
):
    """
    Resume training from the latest checkpoint or a specific checkpoint.

    By default, finds the most recent checkpoint in the checkpoint directory.
    If --config is provided, uses that config's checkpoint_dir.
    If --checkpoint is provided, resumes from that specific checkpoint.

    Examples:
        yagpt resume
        yagpt resume --config configs/optimized-128gb.yaml
        yagpt resume --checkpoint ./checkpoints/checkpoint_iter_5000.pt
        yagpt resume --config configs/optimized-128gb.yaml --checkpoint ./checkpoints/train_val_split/checkpoint_iter_5000.pt
    """
    # Load config if provided, otherwise use default
    if config:
        console.print(f"[bold green]Loading configuration from:[/bold green] {config}")
        train_config = TrainingConfig.from_yaml(str(config))
        checkpoint_dir = Path(train_config.checkpoint_dir)
    else:
        console.print("[bold green]Using default configuration[/bold green]")
        train_config = TrainingConfig()
        checkpoint_dir = Path(train_config.checkpoint_dir)

    # Determine which checkpoint to use
    if checkpoint:
        # User specified a specific checkpoint
        checkpoint_path = checkpoint
        console.print(f"[bold yellow]Resuming from specified checkpoint:[/bold yellow] {checkpoint_path}")
    else:
        # Auto-detect latest checkpoint
        console.print(f"[bold]Searching for checkpoints in:[/bold] {checkpoint_dir}")

        if not checkpoint_dir.exists():
            console.print(f"[bold red]Error:[/bold red] Checkpoint directory not found: {checkpoint_dir}")
            console.print("[yellow]Starting training from scratch instead...[/yellow]\n")
            run_training(train_config)
            return

        checkpoint_path = find_latest_checkpoint(checkpoint_dir)

        if not checkpoint_path:
            console.print(f"[bold yellow]No checkpoints found in {checkpoint_dir}[/bold yellow]")
            console.print("[yellow]Starting training from scratch instead...[/yellow]\n")
            run_training(train_config)
            return

        console.print(f"[bold green]Found latest checkpoint:[/bold green] {checkpoint_path}")

    # Set resume_from in config
    train_config.resume_from = str(checkpoint_path)

    # Show iteration info if we can parse it
    try:
        iteration = int(checkpoint_path.stem.split('_')[-1])
        console.print(f"[bold]Resuming from iteration:[/bold] {iteration:,}\n")
    except:
        console.print()

    run_training(train_config)


@app.command()
def finetune(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to pre-trained checkpoint",
        exists=True,
    ),
    data_dir: Path = typer.Argument(
        ...,
        help="Directory containing fine-tuning data",
        exists=True,
        file_okay=False,
    ),
    max_iters: int = typer.Option(
        10000,
        "--max-iters",
        "-n",
        help="Number of fine-tuning iterations",
    ),
    learning_rate: float = typer.Option(
        1e-4,
        "--learning-rate",
        "--lr",
        help="Learning rate for fine-tuning (lower than pre-training)",
    ),
):
    """
    Fine-tune a pre-trained model on new data.

    Uses a lower learning rate by default (1e-4) compared to pre-training.

    Examples:
        yagpt finetune ./checkpoints/checkpoint_iter_50000.pt ./datasets/my-data
        yagpt finetune ./checkpoints/checkpoint_iter_50000.pt ./datasets/my-data --max-iters 5000 --lr 5e-5
    """
    console.print(f"[bold green]Fine-tuning from:[/bold green] {checkpoint}")
    console.print(f"[bold green]Data directory:[/bold green] {data_dir}")
    console.print(f"[bold]Learning rate:[/bold] {learning_rate}")
    console.print(f"[bold]Max iterations:[/bold] {max_iters:,}\n")

    config = TrainingConfig(
        # Load from checkpoint
        resume_from=str(checkpoint),
        # Fine-tuning data
        data_dir=str(data_dir),
        # Fine-tuning settings
        learning_rate=learning_rate,
        max_iters=max_iters,
        warmup_iters=500,
        # Checkpointing
        checkpoint_interval=500,
    )

    run_training(config)


@app.command()
def generate(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to model checkpoint",
        exists=True,
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Text prompt for generation",
    ),
    max_tokens: int = typer.Option(
        100,
        "--max-tokens",
        "-n",
        help="Maximum number of tokens to generate",
    ),
    temperature: float = typer.Option(
        0.8,
        "--temperature",
        "-t",
        help="Sampling temperature (0.0 = deterministic, higher = more random)",
    ),
    top_k: int = typer.Option(
        40,
        "--top-k",
        help="Top-k sampling parameter",
    ),
    top_p: float = typer.Option(
        0.9,
        "--top-p",
        help="Top-p (nucleus) sampling parameter",
    ),
):
    """
    Generate text from a trained model checkpoint.

    If no prompt is provided via --prompt, shows sample generations with default prompts.

    Examples:
        yagpt generate ./checkpoints/checkpoint_iter_10000.pt
        yagpt generate ./checkpoints/checkpoint_iter_10000.pt --prompt "Once upon a time"
        yagpt generate ./checkpoints/checkpoint_iter_10000.pt -p "The future of AI" -n 200 -t 0.9
    """
    console.print(f"[bold green]Loading model from:[/bold green] {checkpoint}\n")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_from_checkpoint(str(checkpoint), device=device)

    if prompt:
        # Single prompt generation
        console.print(f"[bold]Prompt:[/bold] {prompt}")
        console.print(f"[dim]Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}[/dim]\n")

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )

        console.print(f"[bold cyan]Generated:[/bold cyan]\n{generated}\n")
    else:
        # Sample prompts
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time, in a distant galaxy,",
            "The key to happiness is",
            "In the year 2050,",
        ]

        console.print("[bold]Generating sample completions...[/bold]")
        console.print(f"[dim]Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}[/dim]\n")
        console.print("=" * 70 + "\n")

        for i, p in enumerate(prompts, 1):
            console.print(f"[bold cyan][{i}/{len(prompts)}] Prompt:[/bold cyan] {p}")

            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=p,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )

            console.print(f"[bold green]Generated:[/bold green] {generated}\n")
            console.print("-" * 70 + "\n")


@app.command()
def create_config(
    name: str = typer.Argument(
        ...,
        help="Name for the new config file (without .yaml extension)",
    ),
    template: str = typer.Option(
        "default",
        "--template",
        "-t",
        help="Base template: default, optimized-128gb, medium, long-context, or custom",
    ),
):
    """
    Create a new configuration file from a template.

    Examples:
        yagpt create-config my-config
        yagpt create-config my-config --template optimized-128gb
        yagpt create-config custom-training --template custom
    """
    # Add .yaml extension if not present
    if not name.endswith('.yaml'):
        name += '.yaml'

    template_map = {
        'default': 'configs/default.yaml',
        'optimized-128gb': 'configs/optimized-128gb.yaml',
        'medium': 'configs/medium.yaml',
        'long-context': 'configs/long-context.yaml',
    }

    if template in template_map:
        template_path = Path(template_map[template])
        if not template_path.exists():
            console.print(f"[bold red]Error:[/bold red] Template not found: {template_path}")
            raise typer.Exit(1)

        console.print(f"[bold green]Loading template:[/bold green] {template}")
        with open(template_path, 'r') as f:
            config_data = yaml.safe_load(f)
    elif template == 'custom':
        console.print("[bold green]Creating minimal custom configuration[/bold green]")
        config_data = {
            'model': {
                'name': 'gpt_mini_custom',
                'vocab_size': 100277,
                'n_layer': 12,
                'n_head': 12,
                'n_embd': 768,
                'block_size': 2048,
                'dropout': 0.1,
            },
            'training': {
                'batch_size': 8,
                'gradient_accumulation_steps': 4,
                'max_iters': 100000,
                'learning_rate': 0.0003,
                'weight_decay': 0.1,
                'beta1': 0.9,
                'beta2': 0.95,
                'grad_clip': 1.0,
                'warmup_iters': 2000,
                'lr_decay_iters': 100000,
                'min_lr': 0.00003,
            },
            'evaluation': {
                'eval_interval': 500,
                'eval_iters': 100,
            },
            'logging': {
                'log_interval': 10,
                'backends': ['console', 'csv'],
                'wandb_project': 'yagpt',
            },
            'checkpointing': {
                'checkpoint_dir': './checkpoints',
                'checkpoint_interval': 1000,
                'keep_last_n_checkpoints': 5,
            },
            'data': {
                'data_dir': './datasets/fineweb',
                'num_workers': 4,
            },
            'system': {
                'device': 'cuda',
                'compile': True,
            },
        }
    else:
        console.print(f"[bold red]Error:[/bold red] Unknown template: {template}")
        console.print("[yellow]Available templates:[/yellow] default, optimized-128gb, medium, long-context, custom")
        raise typer.Exit(1)

    # Save config
    output_path = Path('configs') / name
    output_path.parent.mkdir(exist_ok=True)

    console.print(f"[bold green]Saving configuration to:[/bold green] {output_path}")
    with open(output_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)

    console.print("\n[bold green]✓ Configuration created successfully![/bold green]")
    console.print(f"\n[bold]To use this configuration:[/bold]")
    console.print(f"  yagpt train --config {output_path}")


@app.command()
def test():
    """
    Run quick tests to verify installation and basic functionality.

    Tests tokenizer, model creation, forward pass, and generation.
    """
    console.print("[bold]Running YAGPT quick tests...[/bold]\n")

    # Test tokenizer
    console.print("[bold cyan]1. Testing tokenizer...[/bold cyan]")
    tokenizer = GPT4Tokenizer()
    text = "Hello, world! This is a test."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    console.print(f"   Text: {text}")
    console.print(f"   Tokens: {tokens[:10]}... ({len(tokens)} total)")
    console.print(f"   Decoded: {decoded}")
    console.print("   [green]✓ Tokenizer works![/green]\n")

    # Test model
    console.print("[bold cyan]2. Testing model...[/bold cyan]")
    model = create_gpt_mini(
        n_layer=2,  # Small for testing
        n_head=2,
        n_embd=128,
        block_size=128,
    )

    # Test forward pass
    batch = torch.randint(0, tokenizer.vocab_size, (2, 64))
    logits, loss, _ = model(batch, batch)
    console.print(f"   Input shape: {batch.shape}")
    console.print(f"   Output shape: {logits.shape}")
    console.print(f"   Loss: {loss.item():.4f}")
    console.print("   [green]✓ Model works![/green]\n")

    # Test generation
    console.print("[bold cyan]3. Testing generation...[/bold cyan]")
    model.eval()
    start = torch.tensor([[42]])
    generated = model.generate(start, max_new_tokens=10, use_cache=False)
    console.print(f"   Generated tokens: {generated[0].tolist()}")
    console.print("   [green]✓ Generation works![/green]\n")

    console.print("[bold green]✅ All tests passed![/bold green]")


@app.command()
def info():
    """
    Display information about available configs and checkpoints.
    """
    console.print("[bold]YAGPT Information[/bold]\n")

    # Show available configs
    configs_dir = Path("configs")
    if configs_dir.exists():
        config_files = list(configs_dir.glob("*.yaml"))
        if config_files:
            table = Table(title="Available Configurations", show_header=True)
            table.add_column("Config File", style="cyan")
            table.add_column("Path", style="dim")

            for cfg in sorted(config_files):
                table.add_row(cfg.name, str(cfg))

            console.print(table)
            console.print()

    # Show checkpoints
    checkpoints_dir = Path("./checkpoints")
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("**/*.pt"))
        if checkpoints:
            table = Table(title="Available Checkpoints", show_header=True)
            table.add_column("Checkpoint", style="cyan")
            table.add_column("Iteration", style="green")
            table.add_column("Path", style="dim")

            for ckpt in sorted(checkpoints):
                try:
                    iteration = int(ckpt.stem.split('_')[-1])
                    table.add_row(ckpt.name, f"{iteration:,}", str(ckpt))
                except:
                    table.add_row(ckpt.name, "N/A", str(ckpt))

            console.print(table)


if __name__ == "__main__":
    app()
