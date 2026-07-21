#!/usr/bin/env python3
"""
YAGPT Data Preparation CLI.

Commands:
    download-fineweb  Download FineWeb-Edu dataset
    tokenize          Pre-tokenize text data to token shards
    validate          Validate dataset shards
    download-sft      Download and convert SFT dataset to ChatML JSONL
    validate-sft      Validate SFT JSONL file
"""

import glob
import json
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
# SFT dataset downloading
# ---------------------------------------------------------------------------

SFT_DATASETS: dict[str, dict] = {
    "OpenOrca": {
        "repo": "Open-Orca/OpenOrca",
        "format": "openorca",
        "desc": "GPT-4 augmented Flan dataset, 3.2M rows",
    },
    "SlimOrca": {
        "repo": "Open-Orca/SlimOrca",
        "format": "sharegpt",
        "desc": "Curated 500k subset of OpenOrca",
    },
    "Alpaca": {
        "repo": "tatsu-lab/alpaca",
        "format": "alpaca",
        "desc": "52k instruction-following examples from GPT-3.5",
    },
    "UltraChat": {
        "repo": "stingning/ultrachat",
        "format": "sharegpt",
        "desc": "1.5M multi-turn dialogues across diverse topics",
    },
    "Dolly": {
        "repo": "databricks/databricks-dolly-15k",
        "format": "dolly",
        "desc": "15k human-written instruction/response pairs",
    },
    "OASST1": {
        "repo": "OpenAssistant/oasst1",
        "format": "oasst",
        "desc": "66k human-annotated assistant conversation trees",
    },
    "ShareGPT": {
        "repo": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "format": "sharegpt",
        "desc": "Cleaned real ChatGPT conversations, multi-turn",
    },
    "WizardLM": {
        "repo": "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
        "format": "sharegpt",
        "desc": "196k evolved complexity instruction data",
    },
    "Capybara": {
        "repo": "LDJnr/Capybara",
        "format": "capybara",
        "desc": "16k high-quality multi-turn conversations",
    },
    "OpenHermes": {
        "repo": "teknium/OpenHermes-2.5",
        "format": "sharegpt",
        "desc": "1M diverse synthetic GPT-4 conversations",
    },
}


def _convert_openorca(row: dict) -> list[dict[str, str]]:
    """OpenOrca format: {system_prompt, question, response} -> messages."""
    messages: list[dict[str, str]] = []
    if row.get("system_prompt"):
        messages.append({"role": "system", "content": row["system_prompt"]})
    messages.append({"role": "user", "content": row.get("question", "")})
    messages.append({"role": "assistant", "content": row.get("response", "")})
    return messages


def _convert_alpaca(row: dict) -> list[dict[str, str]]:
    """Alpaca format: {instruction, input, output} -> messages."""
    messages: list[dict[str, str]] = []
    user_content = row.get("instruction", "")
    if row.get("input"):
        user_content += "\n\n" + row["input"]
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": row.get("output", "")})
    return messages


def _convert_sharegpt(row: dict) -> list[dict[str, str]]:
    """ShareGPT format: {conversations: [{from, value}]} -> messages."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages: list[dict[str, str]] = []
    convos = row.get("conversations") or row.get("conversation") or []
    for turn in convos:
        role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
        messages.append({"role": role, "content": turn.get("value", "")})
    return messages


def _convert_dolly(row: dict) -> list[dict[str, str]]:
    """Dolly format: {instruction, context, response} -> messages."""
    messages: list[dict[str, str]] = []
    user_content = row.get("instruction", "")
    if row.get("context"):
        user_content += "\n\n" + row["context"]
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": row.get("response", "")})
    return messages


def _convert_oasst(row: dict) -> list[dict[str, str]]:
    """OASST format: flat row with {role, text} -> single message.

    OASST stores one message per row in a tree structure. We convert
    each row to a single turn; tree reconstruction happens at download
    time in do_download_sft via _build_oasst_conversations.
    """
    role_map = {"prompter": "user", "assistant": "assistant"}
    role = role_map.get(row.get("role", ""), "user")
    return [{"role": role, "content": row.get("text", "")}]


def _convert_capybara(row: dict) -> list[dict[str, str]]:
    """Capybara format: {conversation: [{input, output}]} -> messages."""
    messages: list[dict[str, str]] = []
    convos = row.get("conversation") or []
    for turn in convos:
        if turn.get("input"):
            messages.append({"role": "user", "content": turn["input"]})
        if turn.get("output"):
            messages.append({"role": "assistant", "content": turn["output"]})
    return messages


_FORMAT_CONVERTERS = {
    "openorca": _convert_openorca,
    "alpaca": _convert_alpaca,
    "sharegpt": _convert_sharegpt,
    "dolly": _convert_dolly,
    "oasst": _convert_oasst,
    "capybara": _convert_capybara,
}


def _detect_format(row: dict) -> str | None:
    """Auto-detect row format for custom datasets."""
    if "conversations" in row:
        return "sharegpt"
    if "conversation" in row:
        # Distinguish Capybara ({input,output}) from ShareGPT ({from,value})
        convos = row["conversation"]
        if convos and isinstance(convos, list) and isinstance(convos[0], dict):
            if "input" in convos[0]:
                return "capybara"
            return "sharegpt"
    if "instruction" in row and "context" in row:
        return "dolly"
    if "instruction" in row:
        return "alpaca"
    if "system_prompt" in row and "question" in row:
        return "openorca"
    if "message_tree_id" in row:
        return "oasst"
    return None


def _download_oasst(
    ds: object,
    output_path: Path,
    max_rows: int | None,
    on_progress: ProgressCallback | None,
) -> int:
    """Buffer OASST tree-structured data and reconstruct conversations."""
    from collections import defaultdict

    role_map = {"prompter": "user", "assistant": "assistant"}

    # Buffer all messages, grouped by tree
    trees: dict[str, list[dict]] = defaultdict(list)
    children: dict[str | None, list[str]] = defaultdict(list)
    msg_map: dict[str, dict] = {}

    console.print("  Buffering OASST messages...")
    for row in ds:
        msg_id = row.get("message_id", "")
        tree_id = row.get("message_tree_id", "")
        trees[tree_id].append(row)
        children[row.get("parent_id")].append(msg_id)
        msg_map[msg_id] = row

    # For each tree, walk from root to build the longest conversation path
    rows_written = 0
    with open(output_path, "w") as f:
        for tree_id, msgs in trees.items():
            # Find root (parent_id is None)
            roots = [m for m in msgs if m.get("parent_id") is None]
            if not roots:
                continue

            # Walk deepest path from root
            current = roots[0]["message_id"]
            messages: list[dict[str, str]] = []
            while current:
                row = msg_map[current]
                role = role_map.get(row.get("role", ""), "user")
                messages.append({"role": role, "content": row.get("text", "")})
                # Pick highest-ranked child, or first child
                kids = children.get(current, [])
                if not kids:
                    break
                kids.sort(key=lambda k: msg_map[k].get("rank", 999))
                current = kids[0]

            if len(messages) >= 2:
                f.write(json.dumps({"messages": messages}) + "\n")
                rows_written += 1
                if rows_written % 500 == 0 and on_progress:
                    on_progress(rows_written, max_rows or 0, rows_written)
                if max_rows and rows_written >= max_rows:
                    break

    return rows_written


def do_download_sft(
    output_dir: Path,
    dataset: str,
    subset: str | None = None,
    max_rows: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Download an SFT dataset from HuggingFace and convert to ChatML JSONL.

    Returns dict with {rows: int, path: str}.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "'datasets' package required. Install with: pip install datasets"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset info
    if dataset in SFT_DATASETS:
        info = SFT_DATASETS[dataset]
        repo = info["repo"]
        fmt = info["format"]
        name = dataset
    else:
        repo = dataset
        fmt = None  # will auto-detect
        name = dataset.replace("/", "_")

    console.print(f"[bold]Downloading SFT dataset: {repo}[/bold]")
    if subset:
        console.print(f"  Subset: {subset}")

    load_kwargs: dict = {
        "path": repo,
        "split": "train",
        "streaming": True,
    }
    if subset:
        load_kwargs["name"] = subset

    ds = load_dataset(**load_kwargs)

    output_path = output_dir / f"{name}.jsonl"
    rows_written = 0
    converter = _FORMAT_CONVERTERS.get(fmt) if fmt else None

    # OASST is tree-structured (one message per row) — needs buffered reconstruction
    if fmt == "oasst":
        rows_written = _download_oasst(ds, output_path, max_rows, on_progress)
        console.print(f"\n[bold green]Done![/bold green] {rows_written:,} conversations -> {output_path}")
        return {"rows": rows_written, "path": str(output_path)}

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Downloading...", total=None)

        with open(output_path, "w") as f:
            for row in ds:
                # Auto-detect format on first row if needed
                if converter is None:
                    detected = _detect_format(row)
                    if detected:
                        converter = _FORMAT_CONVERTERS[detected]
                        console.print(f"  Auto-detected format: {detected}")
                    else:
                        raise ValueError(
                            f"Cannot detect format for dataset '{dataset}'. "
                            "Fields: " + ", ".join(row.keys())
                        )

                messages = converter(row)
                if messages:
                    f.write(json.dumps({"messages": messages}) + "\n")
                    rows_written += 1

                if rows_written % 1000 == 0:
                    progress.update(task, description=f"Downloaded {rows_written:,} rows")
                    if on_progress:
                        total = max_rows or 0
                        on_progress(rows_written, total, rows_written)

                if max_rows and rows_written >= max_rows:
                    break

    console.print(f"\n[bold green]Done![/bold green] {rows_written:,} rows -> {output_path}")
    return {"rows": rows_written, "path": str(output_path)}


def do_validate_sft(data_path: Path) -> dict:
    """Validate an SFT JSONL file.

    Returns dict with {rows: int, avg_turns: float, issues: list[str]}.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"File '{data_path}' does not exist.")

    console.print(f"[bold]Validating SFT data: {data_path}[/bold]\n")

    total_rows = 0
    total_turns = 0
    issues: list[str] = []

    with open(data_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                issues.append(f"Line {line_num}: invalid JSON")
                continue

            if "messages" not in obj:
                issues.append(f"Line {line_num}: missing 'messages' key")
                continue

            messages = obj["messages"]
            if not isinstance(messages, list):
                issues.append(f"Line {line_num}: 'messages' is not a list")
                continue

            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    issues.append(f"Line {line_num}, msg {i}: not a dict")
                    continue
                if "role" not in msg:
                    issues.append(f"Line {line_num}, msg {i}: missing 'role'")
                if "content" not in msg:
                    issues.append(f"Line {line_num}, msg {i}: missing 'content'")

            total_rows += 1
            total_turns += len(messages)

    avg_turns = total_turns / max(total_rows, 1)

    from rich.table import Table

    result = Table(title="SFT Data Validation")
    result.add_column("Metric", style="cyan")
    result.add_column("Value", style="green")
    result.add_row("Rows", f"{total_rows:,}")
    result.add_row("Avg turns/row", f"{avg_turns:.1f}")
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

    return {"rows": total_rows, "avg_turns": avg_turns, "issues": issues}


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


@app.command()
def download_sft(
    output_dir: Path = typer.Argument(..., help="Output directory for JSONL file"),
    dataset: str = typer.Option(
        "OpenOrca",
        "--dataset", "-d",
        help="Dataset name from registry or HuggingFace repo (e.g., user/repo)",
    ),
    subset: Optional[str] = typer.Option(
        None,
        "--subset", "-s",
        help="HuggingFace dataset config/subset name",
    ),
    max_rows: Optional[int] = typer.Option(
        None,
        "--max-rows",
        help="Maximum rows to download (None = all)",
    ),
) -> None:
    """Download and convert an SFT dataset to ChatML JSONL."""
    try:
        do_download_sft(output_dir, dataset, subset, max_rows)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate_sft(
    data_path: Path = typer.Argument(..., help="Path to SFT JSONL file"),
) -> None:
    """Validate an SFT JSONL file."""
    try:
        do_validate_sft(data_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
