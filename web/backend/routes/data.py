"""POST /api/data/{download,tokenize,validate} — data preparation routes."""

from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import pipeline

router = APIRouter()

KNOWN_SUBSETS = ["sample-10BT", "sample-100BT", "CC-MAIN-2024-10"]


@router.get("/status")
async def data_status() -> dict:
    """Return download/tokenized shard counts for each known subset."""
    result: dict[str, dict[str, int]] = {}
    for subset in KNOWN_SUBSETS:
        raw_dir = Path(f"./data/raw/{subset}")
        tok_dir = Path(f"./data/tokenized/{subset}")
        raw_shards = len(list(raw_dir.glob("*.parquet"))) if raw_dir.is_dir() else 0
        tok_shards = len(list(tok_dir.glob("*.parquet"))) if tok_dir.is_dir() else 0
        result[subset] = {"raw_shards": raw_shards, "tokenized_shards": tok_shards}
    return result


@router.get("/dirs")
async def data_dirs() -> list[dict]:
    """Return directories that contain tokenized parquet data."""
    results: list[dict] = []
    base = Path("./data/tokenized")
    if base.is_dir():
        for sub in sorted(base.iterdir()):
            if sub.is_dir():
                shards = list(sub.glob("*.parquet"))
                if shards:
                    results.append({
                        "path": str(sub),
                        "name": sub.name,
                        "shards": len(shards),
                    })
    return results


class DownloadRequest(BaseModel):
    output_dir: str = "./data/raw"
    subset: str = "sample-10BT"
    num_shards: int = 100
    max_rows: int | None = None


class TokenizeRequest(BaseModel):
    input_dir: str = "./data/raw"
    output_dir: str = "./data/tokenized"
    encoding: str = "gpt2"
    max_seq_len: int = 2048


class ValidateRequest(BaseModel):
    data_dir: str = "./data/tokenized"


@router.post("/download")
async def download_data(req: DownloadRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A job is already running")

    # Skip if already downloaded enough shards
    raw_dir = Path(req.output_dir)
    existing = len(list(raw_dir.glob("*.parquet"))) if raw_dir.is_dir() else 0
    if existing >= req.num_shards:
        pipeline.broadcast_status("data", "done")
        return {"status": "skipped", "message": f"Already have {existing} shards in {req.output_dir}"}

    pipeline.broadcast_status("data", "running")

    def run() -> None:
        try:
            from scripts.prepare_data import do_download_fineweb

            def on_progress(current: int, total: int, rows: int) -> None:
                pipeline.broadcast_status(
                    "data", "running",
                    progress=current, total=total, rows=rows,
                    message=f"Shard {current}/{total} ({rows:,} rows)",
                )

            do_download_fineweb(
                output_dir=Path(req.output_dir),
                subset=req.subset,
                num_shards=req.num_shards,
                max_rows=req.max_rows,
                on_progress=on_progress,
            )
            pipeline.broadcast_status("data", "done")
        except Exception as e:
            pipeline.broadcast_status("data", "error", error=str(e))
            traceback.print_exc()

    thread = threading.Thread(target=run, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started"}


@router.post("/tokenize")
async def tokenize_data(req: TokenizeRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A job is already running")

    pipeline.broadcast_status("data", "running")

    def run() -> None:
        try:
            from scripts.prepare_data import do_tokenize

            def on_progress(current: int, total: int, tokens: int) -> None:
                pipeline.broadcast_status(
                    "data", "running",
                    progress=current, total=total, tokens=tokens,
                    message=f"Shard {current}/{total} ({tokens:,} tokens)",
                )

            do_tokenize(
                input_dir=Path(req.input_dir),
                output_dir=Path(req.output_dir),
                encoding=req.encoding,
                max_seq_len=req.max_seq_len,
                on_progress=on_progress,
            )
            pipeline.broadcast_status("data", "done")
        except Exception as e:
            pipeline.broadcast_status("data", "error", error=str(e))
            traceback.print_exc()

    thread = threading.Thread(target=run, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started"}


@router.post("/validate")
async def validate_data(req: ValidateRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A job is already running")

    pipeline.broadcast_status("data", "running")

    def run() -> None:
        try:
            from scripts.prepare_data import do_validate
            do_validate(data_dir=Path(req.data_dir))
            pipeline.broadcast_status("data", "done")
        except Exception as e:
            pipeline.broadcast_status("data", "error", error=str(e))
            traceback.print_exc()

    thread = threading.Thread(target=run, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started"}


# ---------------------------------------------------------------------------
# SFT data routes
# ---------------------------------------------------------------------------


class SFTDownloadRequest(BaseModel):
    output_dir: str = "./data/sft"
    dataset: str = "OpenOrca"
    subset: str | None = None
    max_rows: int | None = None


class SFTValidateRequest(BaseModel):
    data_path: str = "./data/sft/OpenOrca.jsonl"


@router.get("/sft/datasets")
async def sft_datasets() -> dict:
    """Return available SFT datasets and any already-downloaded JSONL files."""
    from scripts.prepare_data import SFT_DATASETS

    available = {k: {"repo": v["repo"]} for k, v in SFT_DATASETS.items()}

    downloaded: list[dict] = []
    sft_dir = Path("./data/sft")
    if sft_dir.is_dir():
        for f in sorted(sft_dir.glob("*.jsonl")):
            rows = sum(1 for line in open(f) if line.strip())
            downloaded.append({
                "name": f.stem,
                "path": str(f),
                "rows": rows,
            })

    return {"available": available, "downloaded": downloaded}


@router.post("/sft/download")
async def sft_download(req: SFTDownloadRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A job is already running")

    pipeline.broadcast_status("data", "running")

    def run() -> None:
        try:
            from scripts.prepare_data import do_download_sft

            def on_progress(current: int, total: int, rows: int) -> None:
                pipeline.broadcast_status(
                    "data", "running",
                    progress=current, total=max(total, 1), rows=rows,
                    message=f"Downloaded {rows:,} rows",
                )

            do_download_sft(
                output_dir=Path(req.output_dir),
                dataset=req.dataset,
                subset=req.subset,
                max_rows=req.max_rows,
                on_progress=on_progress,
            )
            pipeline.broadcast_status("data", "done")
        except Exception as e:
            pipeline.broadcast_status("data", "error", error=str(e))
            traceback.print_exc()

    thread = threading.Thread(target=run, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started"}


@router.post("/sft/validate")
async def sft_validate(req: SFTValidateRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A job is already running")

    pipeline.broadcast_status("data", "running")

    def run() -> None:
        try:
            from scripts.prepare_data import do_validate_sft
            result = do_validate_sft(data_path=Path(req.data_path))
            pipeline.broadcast_status(
                "data", "done",
                rows=result["rows"],
                avg_turns=result["avg_turns"],
                issues=len(result["issues"]),
            )
        except Exception as e:
            pipeline.broadcast_status("data", "error", error=str(e))
            traceback.print_exc()

    thread = threading.Thread(target=run, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started"}
