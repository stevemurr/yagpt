#!/usr/bin/env python3
"""
YAGPT Command Line Interface.

Commands:
    train     Train a model from scratch
    generate  Generate text from a checkpoint
    info      Show model/checkpoint information
"""

from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="yagpt",
    help="YAGPT - Yet Another GPT",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to training config YAML",
        exists=True,
    ),
):
    """Train a model from scratch or resume from checkpoint."""
    from yagpt import Trainer, TrainConfig, Tokenizer, create_dataloader
    from yagpt.training import (
        LoggingCallback,
        CheckpointCallback,
        EvalCallback,
        SampleCallback,
        WandbCallback,
    )

    console.print(f"[bold]Loading config:[/bold] {config}")
    cfg = TrainConfig.from_yaml(config)

    # Create tokenizer
    tokenizer = Tokenizer(cfg.tokenizer)
    console.print(f"[bold]Tokenizer:[/bold] {tokenizer}")

    # Create data loaders
    console.print(f"[bold]Train data:[/bold] {cfg.train_data_dir}")
    train_loader = create_dataloader(
        cfg.train_data_dir,
        seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size * cfg.grad_accum_steps,
        tokenizer=tokenizer,
        num_workers=cfg.num_workers,
    )

    val_loader = None
    if cfg.val_data_dir:
        console.print(f"[bold]Val data:[/bold] {cfg.val_data_dir}")
        val_loader = create_dataloader(
            cfg.val_data_dir,
            seq_len=cfg.max_seq_len,
            batch_size=cfg.batch_size,
            tokenizer=tokenizer,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

    # Setup callbacks
    callbacks = [
        LoggingCallback(cfg.log_interval),
        CheckpointCallback(cfg.checkpoint_dir, cfg.checkpoint_interval, cfg.keep_checkpoints),
    ]

    if val_loader:
        callbacks.append(EvalCallback(cfg.eval_interval, cfg.eval_steps))

    if cfg.sample_interval > 0:
        callbacks.append(SampleCallback(cfg.sample_interval))

    if cfg.use_wandb:
        callbacks.append(WandbCallback(cfg.wandb_project, cfg.wandb_run_name))

    # Create trainer and run
    trainer = Trainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
    )

    trainer.train()


@app.command()
def generate(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to model checkpoint",
        exists=True,
    ),
    prompt: str = typer.Option(
        "Once upon a time,",
        "--prompt", "-p",
        help="Text prompt",
    ),
    max_tokens: int = typer.Option(
        100,
        "--max-tokens", "-n",
        help="Maximum tokens to generate",
    ),
    temperature: float = typer.Option(
        0.8,
        "--temperature", "-t",
        help="Sampling temperature",
    ),
    top_k: Optional[int] = typer.Option(
        40,
        "--top-k",
        help="Top-k sampling",
    ),
):
    """Generate text from a trained model."""
    from yagpt import GPT, GPTConfig, Tokenizer

    console.print(f"[bold]Loading checkpoint:[/bold] {checkpoint}")

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    # Create tokenizer
    tokenizer = Tokenizer(cfg.get("tokenizer", "gpt2"))

    # Create model
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg.get("n_kv_heads"),
        dim=cfg["dim"],
        hidden_dim=cfg.get("hidden_dim"),
        max_seq_len=cfg["max_seq_len"],
    )

    model = GPT(model_config)
    model.load_state_dict(ckpt["model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    console.print(f"[bold]Model:[/bold] {model.num_parameters()/1e6:.1f}M parameters")
    console.print()

    # Generate
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)

    console.print(f"[cyan]Prompt:[/cyan] {prompt}")
    console.print()

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    text = tokenizer.decode(output[0].tolist())
    console.print(f"[green]{text}[/green]")


@app.command()
def info(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to model checkpoint",
        exists=True,
    ),
):
    """Show information about a checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu")

    table = Table(title=f"Checkpoint: {checkpoint.name}")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    # Basic info
    table.add_row("Step", str(ckpt.get("step", "N/A")))
    table.add_row("Loss", f"{ckpt.get('loss', 'N/A'):.4f}" if "loss" in ckpt else "N/A")
    if "val_loss" in ckpt:
        table.add_row("Val Loss", f"{ckpt['val_loss']:.4f}")

    # Config
    cfg = ckpt.get("config", {})
    table.add_row("", "")  # Separator
    table.add_row("[bold]Model Config[/bold]", "")
    table.add_row("  Layers", str(cfg.get("n_layers", "N/A")))
    table.add_row("  Heads", str(cfg.get("n_heads", "N/A")))
    table.add_row("  Dim", str(cfg.get("dim", "N/A")))
    table.add_row("  Max Seq Len", str(cfg.get("max_seq_len", "N/A")))

    console.print(table)


@app.command()
def count(
    data_dir: Path = typer.Argument(
        ...,
        help="Directory containing parquet shards",
        exists=True,
    ),
):
    """Count tokens in a dataset."""
    import glob
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(str(data_dir / "*.parquet")))
    console.print(f"Found {len(shards)} shards")

    total_tokens = 0
    total_rows = 0

    for shard in shards:
        table = pq.read_table(shard)
        total_rows += len(table)

        if "tokens" in table.column_names:
            for batch in table.to_batches():
                for tokens in batch["tokens"]:
                    total_tokens += len(tokens.as_py())

    console.print(f"Total rows: {total_rows:,}")
    if total_tokens > 0:
        console.print(f"Total tokens: {total_tokens:,}")
        console.print(f"Avg tokens/row: {total_tokens/total_rows:.1f}")


if __name__ == "__main__":
    app()
