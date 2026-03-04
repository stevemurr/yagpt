#!/usr/bin/env python3
"""
YAGPT Data Preparation CLI.

Commands:
    download-fineweb  Download FineWeb-Edu dataset
    tokenize          Pre-tokenize text data to token shards
    validate          Validate dataset shards
"""

import glob
from pathlib import Path
from typing import Callable, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="yagpt-data",
    help="YAGPT Data Preparation Tools",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Core functions (called by both CLI and web routes)
# ---------------------------------------------------------------------------


ProgressCallback = Callable[[int, int, int], None]  # (current, total, rows)


def do_download_fineweb(
    output_dir: Path,
    subset: str = "sample-10BT",
    num_shards: int = 100,
    max_rows: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Download FineWeb-Edu dataset and shard to parquet files.

    Returns dict with {shards: int, rows: int}.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "'datasets' package required. Install with: pip install datasets"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect existing shards to support resuming interrupted downloads
    existing_shards = sorted(output_dir.glob("shard_*.parquet"))
    start_shard = len(existing_shards)

    if start_shard >= num_shards:
        console.print(f"[bold green]Already have {start_shard} shards, nothing to download.[/bold green]")
        return {"shards": start_shard, "rows": 0}

    console.print(f"[bold]Downloading FineWeb-Edu ({subset})[/bold]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Shards: {num_shards} ({start_shard} existing, {num_shards - start_shard} remaining)")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset,
        split="train",
        streaming=True,
    )

    shard_idx = start_shard
    buffer: list[dict] = []
    rows_written = 0
    rows_per_shard = (max_rows or 1_000_000) // num_shards
    # Skip rows that correspond to already-written shards
    rows_to_skip = start_shard * rows_per_shard

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Downloading...", total=None)

        for row in ds:
            text = row.get("text", "")
            if not text:
                continue

            if rows_to_skip > 0:
                rows_to_skip -= 1
                continue

            buffer.append({"text": text})
            rows_written += 1

            if len(buffer) >= rows_per_shard:
                _write_shard(output_dir, shard_idx, buffer)
                shard_idx += 1
                buffer = []
                progress.update(task, description=f"Written shard {shard_idx}/{num_shards}")
                if on_progress:
                    on_progress(shard_idx, num_shards, rows_written)

                if shard_idx >= num_shards:
                    break

            if max_rows and rows_written >= max_rows:
                break

    if buffer and shard_idx < num_shards:
        _write_shard(output_dir, shard_idx, buffer)
        shard_idx += 1

    console.print(f"\n[bold green]Done![/bold green] {shard_idx} shards, {rows_written:,} new rows")
    return {"shards": shard_idx, "rows": rows_written}


def _write_shard(output_dir: Path, shard_idx: int, rows: list[dict]) -> None:
    """Write a list of rows to a parquet shard."""
    table = pa.table({"text": [r["text"] for r in rows]})
    path = output_dir / f"shard_{shard_idx:05d}.parquet"
    pq.write_table(table, path)


def do_tokenize(
    input_dir: Path,
    output_dir: Path,
    encoding: str = "gpt2",
    max_seq_len: int = 2048,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Pre-tokenize text parquet shards to token parquet shards.

    Returns dict with {shards: int, total_tokens: int}.
    """
    from yagpt.tokenizer import Tokenizer

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory '{input_dir}' does not exist. Download the dataset first."
        )

    shards = sorted(glob.glob(str(input_dir / "*.parquet")))
    if not shards:
        raise FileNotFoundError(
            f"No parquet files found in '{input_dir}'. Download the dataset first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(encoding)
    console.print(f"[bold]Tokenizer:[/bold] {tokenizer}")
    console.print(f"[bold]Input shards:[/bold] {len(shards)}")

    total_tokens = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Tokenizing...", total=len(shards))

        for i, shard_path in enumerate(shards):
            table = pq.read_table(shard_path)

            if "text" not in table.column_names:
                console.print(f"[yellow]Skipping {shard_path}: no 'text' column[/yellow]")
                continue

            all_tokens = []
            for batch in table.to_batches(max_chunksize=1000):
                for j in range(len(batch)):
                    text = batch["text"][j].as_py()
                    tokens = tokenizer.encode(text)
                    all_tokens.append(tokens)
                    total_tokens += len(tokens)

            out_table = pa.table({"tokens": all_tokens})
            out_path = output_dir / f"shard_{i:05d}.parquet"
            pq.write_table(out_table, out_path)

            progress.update(task, completed=i + 1, description=f"Shard {i+1}/{len(shards)}")
            if on_progress:
                on_progress(i + 1, len(shards), total_tokens)

    console.print(f"\n[bold green]Done![/bold green] {total_tokens:,} tokens across {len(shards)} shards")
    console.print(f"  Avg tokens/shard: {total_tokens // max(len(shards), 1):,}")
    return {"shards": len(shards), "total_tokens": total_tokens}


def do_validate(data_dir: Path) -> dict:
    """Validate dataset shards: check format, count tokens, detect issues.

    Returns dict with validation summary.
    """
    from rich.table import Table

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")

    shards = sorted(glob.glob(str(data_dir / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No parquet files found in '{data_dir}'.")

    console.print(f"[bold]Validating {len(shards)} shards in {data_dir}[/bold]\n")

    total_rows = 0
    total_tokens = 0
    issues: list[str] = []
    has_tokens = False
    has_text = False

    for shard_path in shards:
        try:
            table = pq.read_table(shard_path)
            n_rows = len(table)
            total_rows += n_rows

            if n_rows == 0:
                issues.append(f"{Path(shard_path).name}: empty shard")

            if "tokens" in table.column_names:
                has_tokens = True
                for batch in table.to_batches(max_chunksize=1000):
                    for i in range(len(batch)):
                        tokens = batch["tokens"][i].as_py()
                        total_tokens += len(tokens)
                        if len(tokens) == 0:
                            issues.append(f"{Path(shard_path).name}: empty token sequence")

            elif "text" in table.column_names:
                has_text = True
            else:
                issues.append(f"{Path(shard_path).name}: no 'tokens' or 'text' column")

        except Exception as e:
            issues.append(f"{Path(shard_path).name}: {e}")

    fmt = "pre-tokenized" if has_tokens else "text" if has_text else "unknown"

    result = Table(title="Dataset Validation")
    result.add_column("Metric", style="cyan")
    result.add_column("Value", style="green")
    result.add_row("Shards", str(len(shards)))
    result.add_row("Total rows", f"{total_rows:,}")
    result.add_row("Format", fmt)
    if total_tokens > 0:
        result.add_row("Total tokens", f"{total_tokens:,}")
        result.add_row("Avg tokens/row", f"{total_tokens / max(total_rows, 1):.1f}")
        result.add_row("Approx size", f"{total_tokens / 1e9:.2f}B tokens")
    result.add_row("Issues", str(len(issues)))
    console.print(result)

    if issues:
        console.print("\n[yellow]Issues found:[/yellow]")
        for issue in issues[:20]:
            console.print(f"  - {issue}")
        if len(issues) > 20:
            console.print(f"  ... and {len(issues) - 20} more")
    else:
        console.print("\n[bold green]All checks passed![/bold green]")

    return {
        "shards": len(shards),
        "total_rows": total_rows,
        "total_tokens": total_tokens,
        "format": fmt,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Typer CLI wrappers
# ---------------------------------------------------------------------------


@app.command()
def download_fineweb(
    output_dir: Path = typer.Argument(..., help="Output directory for parquet shards"),
    subset: str = typer.Option(
        "sample-10BT",
        "--subset", "-s",
        help="FineWeb-Edu subset (e.g., sample-10BT, sample-100BT)",
    ),
    num_shards: int = typer.Option(
        100,
        "--num-shards", "-n",
        help="Number of output shards",
    ),
    max_rows: Optional[int] = typer.Option(
        None,
        "--max-rows",
        help="Maximum rows to download (None = all)",
    ),
) -> None:
    """Download FineWeb-Edu dataset and shard to parquet files."""
    try:
        do_download_fineweb(output_dir, subset, num_shards, max_rows)
    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def tokenize(
    input_dir: Path = typer.Argument(..., help="Input directory with text parquet shards"),
    output_dir: Path = typer.Argument(..., help="Output directory for tokenized shards"),
    encoding: str = typer.Option("gpt2", "--encoding", "-e", help="Tokenizer encoding"),
    max_seq_len: int = typer.Option(2048, "--max-seq-len", help="Discard documents shorter than this"),
) -> None:
    """Pre-tokenize text parquet shards to token parquet shards."""
    try:
        do_tokenize(input_dir, output_dir, encoding, max_seq_len)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    data_dir: Path = typer.Argument(..., help="Directory with parquet shards"),
) -> None:
    """Validate dataset shards: check format, count tokens, detect issues."""
    try:
        do_validate(data_dir)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
