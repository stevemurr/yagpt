"""GET /api/checkpoints — list and inspect checkpoints."""

from __future__ import annotations

import os

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class CheckpointInfo(BaseModel):
    path: str
    filename: str
    step: int | None = None
    loss: float | None = None
    val_loss: float | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    dim: int | None = None
    max_seq_len: int | None = None
    size_mb: float | None = None


@router.get("", response_model=list[CheckpointInfo])
async def list_checkpoints(directory: str = "./checkpoints") -> list[CheckpointInfo]:
    if not os.path.isdir(directory):
        return []

    results = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(directory, fname)
        info = _inspect_checkpoint(path, fname)
        if info:
            results.append(info)
    return results


@router.get("/inspect", response_model=CheckpointInfo)
async def inspect_checkpoint(path: str) -> CheckpointInfo:
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {path}")
    info = _inspect_checkpoint(path, os.path.basename(path))
    if not info:
        raise HTTPException(status_code=400, detail="Could not read checkpoint")
    return info


def _inspect_checkpoint(path: str, filename: str) -> CheckpointInfo | None:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return CheckpointInfo(
            path=path,
            filename=filename,
            step=ckpt.get("step"),
            loss=ckpt.get("loss"),
            val_loss=ckpt.get("val_loss"),
            n_layers=cfg.get("n_layers"),
            n_heads=cfg.get("n_heads"),
            dim=cfg.get("dim"),
            max_seq_len=cfg.get("max_seq_len"),
            size_mb=round(size_mb, 1),
        )
    except Exception:
        return None
