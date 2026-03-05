"""POST /api/sft/{start,stop} — supervised fine-tuning routes."""

from __future__ import annotations

import threading
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import pipeline
from ..callbacks import WebSocketCallback
from .. import db

router = APIRouter()


class SFTStartRequest(BaseModel):
    checkpoint: str
    config: dict[str, Any] = {}
    experiment_id: int | None = None


@router.post("/start")
async def start_sft(req: SFTStartRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A training job is already running")

    pipeline.stop_event.clear()
    pipeline.metrics["sft"].clear()

    # Create run record if experiment_id provided
    run_id: int | None = None
    if req.experiment_id:
        try:
            run = await db.create_run(req.experiment_id, "sft", {**req.config, "checkpoint": req.checkpoint})
            run_id = run["id"]
            pipeline.current_experiment_id = req.experiment_id
            pipeline.current_run_id = run_id
        except Exception:
            pass

    def run_sft() -> None:
        try:
            from yagpt.sft.config import SFTConfig
            from yagpt.sft.trainer import SFTTrainer
            from yagpt.sft.dataset import create_sft_dataloader
            from yagpt.tokenizer import Tokenizer

            cfg = SFTConfig(**{**req.config, "base_checkpoint": req.checkpoint})
            tokenizer = Tokenizer(cfg.tokenizer)

            train_loader = create_sft_dataloader(
                data_path=cfg.data_path,
                tokenizer=tokenizer,
                max_seq_len=cfg.max_seq_len,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

            ws_cb = WebSocketCallback(pipeline, "sft")
            trainer = SFTTrainer(config=cfg, train_loader=train_loader, callbacks=[ws_cb])
            trainer.train()

            pipeline.model = trainer.model
            pipeline.model.eval()
            pipeline.tokenizer = tokenizer

            if not pipeline.stop_event.is_set():
                pipeline.broadcast_status("sft", "done")

            if run_id:
                metrics = list(pipeline.metrics["sft"])
                summary = {"final_loss": metrics[-1]["loss"]} if metrics else {}
                try:
                    pipeline.run_async(db.complete_run(run_id, "completed", summary))
                except Exception:
                    pass
        except Exception as e:
            pipeline.broadcast_status("sft", "error", error=str(e))
            traceback.print_exc()
            if run_id:
                try:
                    pipeline.run_async(db.complete_run(run_id, "failed", {"error": str(e)}))
                except Exception:
                    pass

    thread = threading.Thread(target=run_sft, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started"}


@router.post("/stop")
async def stop_sft() -> dict:
    if not pipeline.is_busy():
        raise HTTPException(status_code=400, detail="No training job is running")
    pipeline.stop_event.set()
    return {"status": "stopping"}


@router.get("/status")
async def sft_status() -> dict:
    metrics = pipeline.metrics["sft"]
    step = metrics[-1]["step"] if metrics else None
    return {"status": pipeline.stage_status["sft"], "step": step}


@router.get("/metrics")
async def sft_metrics(last_n: int = 100) -> list[dict]:
    return list(pipeline.metrics["sft"])[-last_n:]
