"""POST /api/pretrain/{start,stop}, GET /api/pretrain/{status,metrics,config/schema}."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import pipeline
from ..callbacks import WebSocketCallback

router = APIRouter()


class PretrainStartRequest(BaseModel):
    config: dict[str, Any] = {}


class StatusResponse(BaseModel):
    status: str
    step: int | None = None
    max_steps: int | None = None
    error: str | None = None


@router.post("/start")
async def start_pretrain(req: PretrainStartRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A training job is already running")

    pipeline.stop_event.clear()
    pipeline.metrics["pretrain"].clear()
    pipeline.clear_model()

    def run_training() -> None:
        try:
            from yagpt.training.config import TrainConfig
            from yagpt.training.trainer import Trainer
            from yagpt.data.dataloader import create_dataloader
            from yagpt.training.callbacks import (
                LoggingCallback, CheckpointCallback, EvalCallback, SampleCallback,
            )
            from yagpt.training.mfu import MFUCallback

            config = TrainConfig(**req.config)
            from yagpt.tokenizer import Tokenizer
            tokenizer = Tokenizer(config.tokenizer)

            train_loader = create_dataloader(
                data_dir=config.train_data_dir,
                seq_len=config.max_seq_len,
                batch_size=config.batch_size * config.grad_accum_steps,
                tokenizer=tokenizer,
                shuffle=True,
                num_workers=config.num_workers,
                seed=config.seed,
            )
            val_loader = None
            if config.val_data_dir and Path(config.val_data_dir).is_dir():
                val_loader = create_dataloader(
                    data_dir=config.val_data_dir,
                    seq_len=config.max_seq_len,
                    batch_size=config.batch_size,
                    tokenizer=tokenizer,
                    shuffle=False,
                    num_workers=config.num_workers,
                )

            ws_cb = WebSocketCallback(pipeline, "pretrain")
            callbacks = [
                ws_cb,
                LoggingCallback(log_interval=config.log_interval),
                CheckpointCallback(
                    checkpoint_dir=config.checkpoint_dir,
                    interval=config.checkpoint_interval,
                    keep_last=config.keep_checkpoints,
                ),
            ]
            if val_loader:
                callbacks.append(EvalCallback(
                    eval_interval=config.eval_interval,
                    eval_steps=config.eval_steps,
                ))
            callbacks.append(SampleCallback(interval=config.sample_interval))

            trainer = Trainer(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                callbacks=callbacks,
            )

            # Add MFU callback now that model exists
            base_model = trainer.model._orig_mod if hasattr(trainer.model, "_orig_mod") else trainer.model
            callbacks.append(MFUCallback(num_params=base_model.num_parameters()))

            trainer.train()

            # Store model for downstream use
            pipeline.model = trainer.model
            pipeline.model.eval()
            from yagpt.tokenizer import Tokenizer
            pipeline.tokenizer = Tokenizer(config.tokenizer)

            if not pipeline.stop_event.is_set():
                pipeline.broadcast_status("pretrain", "done")
        except Exception as e:
            pipeline.broadcast_status("pretrain", "error", error=str(e))
            traceback.print_exc()

    thread = threading.Thread(target=run_training, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started"}


@router.post("/stop")
async def stop_pretrain() -> dict:
    if not pipeline.is_busy():
        raise HTTPException(status_code=400, detail="No training job is running")
    pipeline.stop_event.set()
    return {"status": "stopping"}


@router.get("/status", response_model=StatusResponse)
async def pretrain_status() -> StatusResponse:
    metrics = pipeline.metrics["pretrain"]
    step = metrics[-1]["step"] if metrics else None
    return StatusResponse(
        status=pipeline.stage_status["pretrain"],
        step=step,
    )


@router.get("/metrics")
async def pretrain_metrics(last_n: int = 100) -> list[dict]:
    metrics = pipeline.metrics["pretrain"]
    return list(metrics)[-last_n:]


@router.get("/config/schema")
async def pretrain_config_schema() -> dict:
    """Return the TrainConfig fields with types and defaults for the UI."""
    from yagpt.training.config import TrainConfig
    import dataclasses
    fields = {}
    for f in dataclasses.fields(TrainConfig):
        fields[f.name] = {
            "type": f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type)),
            "default": f.default if f.default is not dataclasses.MISSING else None,
        }
    return fields
