"""POST /api/alignment/{start,stop} — DPO / SimPO / GRPO training routes."""

from __future__ import annotations

import threading
import traceback
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import pipeline
from ..callbacks import WebSocketCallback

router = APIRouter()


class AlignStartRequest(BaseModel):
    checkpoint: str | None = None
    data_path: str
    method: str = "dpo"  # dpo | simpo | grpo
    beta: float = 0.1
    lr: float = 1e-6
    max_steps: int = 1000
    batch_size: int = 4
    gamma: float = 0.5  # SimPO
    group_size: int = 4  # GRPO
    output_dir: str = "./checkpoints/align"


@router.post("/start")
async def start_alignment(req: AlignStartRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A training job is already running")

    # Load checkpoint if needed
    if req.checkpoint and req.checkpoint != pipeline.checkpoint_path:
        try:
            pipeline.load_checkpoint(req.checkpoint)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}")

    if pipeline.model is None or pipeline.tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    pipeline.stop_event.clear()
    pipeline.metrics["align"].clear()

    def run_alignment() -> None:
        try:
            from yagpt.alignment.dataset import PreferenceDataset, preference_collate_fn
            from torch.utils.data import DataLoader

            device = next(pipeline.model.parameters()).device
            dataset = PreferenceDataset(req.data_path, pipeline.tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=req.batch_size,
                shuffle=True,
                collate_fn=preference_collate_fn,
            )

            ws_cb = WebSocketCallback(pipeline, "align")

            if req.method == "dpo":
                from yagpt.alignment.dpo import DPOTrainer
                trainer = DPOTrainer(
                    model=pipeline.model,
                    beta=req.beta,
                    lr=req.lr,
                    max_steps=req.max_steps,
                    device=device,
                    callbacks=[ws_cb],
                )
            elif req.method == "simpo":
                from yagpt.alignment.simpo import SimPOTrainer
                trainer = SimPOTrainer(
                    model=pipeline.model,
                    beta=req.beta,
                    gamma=req.gamma,
                    lr=req.lr,
                    max_steps=req.max_steps,
                    device=device,
                    callbacks=[ws_cb],
                )
            elif req.method == "grpo":
                from yagpt.alignment.grpo import GRPOTrainer
                trainer = GRPOTrainer(
                    model=pipeline.model,
                    tokenizer=pipeline.tokenizer,
                    beta=req.beta,
                    lr=req.lr,
                    max_steps=req.max_steps,
                    group_size=req.group_size,
                    device=device,
                    callbacks=[ws_cb],
                )
            else:
                raise ValueError(f"Unknown method: {req.method}")

            trainer.train(dataloader)
            trainer.save(f"{req.output_dir}/{req.method}_final.pt")

            if not pipeline.stop_event.is_set():
                pipeline.broadcast_status("align", "done")
        except Exception as e:
            pipeline.broadcast_status("align", "error", error=str(e))
            traceback.print_exc()

    thread = threading.Thread(target=run_alignment, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started", "method": req.method}


@router.post("/stop")
async def stop_alignment() -> dict:
    if not pipeline.is_busy():
        raise HTTPException(status_code=400, detail="No training job is running")
    pipeline.stop_event.set()
    return {"status": "stopping"}


@router.get("/status")
async def alignment_status() -> dict:
    metrics = pipeline.metrics["align"]
    step = metrics[-1]["step"] if metrics else None
    return {"status": pipeline.stage_status["align"], "step": step}


@router.get("/metrics")
async def alignment_metrics(last_n: int = 100) -> list[dict]:
    return list(pipeline.metrics["align"])[-last_n:]
