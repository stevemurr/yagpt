"""POST /api/eval/run — run lm-eval-harness benchmarks."""

from __future__ import annotations

import threading
import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import pipeline

router = APIRouter()


class EvalRequest(BaseModel):
    checkpoint: str | None = None
    tasks: str = "hellaswag"
    batch_size: int = 8
    num_fewshot: int = 0


@router.post("/run")
async def run_eval(req: EvalRequest) -> dict:
    if pipeline.is_busy():
        raise HTTPException(status_code=409, detail="A job is already running")

    if req.checkpoint and req.checkpoint != pipeline.checkpoint_path:
        try:
            pipeline.load_checkpoint(req.checkpoint)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}")

    if pipeline.model is None or pipeline.tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    pipeline.stop_event.clear()
    pipeline.broadcast_status("eval", "running")

    def run() -> None:
        try:
            from yagpt.eval.harness import evaluate_model
            import torch

            device = next(pipeline.model.parameters()).device
            results = evaluate_model(
                model=pipeline.model,
                tokenizer=pipeline.tokenizer,
                tasks=req.tasks,
                device=device,
                batch_size=req.batch_size,
                num_fewshot=req.num_fewshot,
            )

            # Extract per-task metrics
            task_results = {}
            if "results" in results:
                for task_name, metrics in results["results"].items():
                    task_results[task_name] = {
                        k: v for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    }

            pipeline.eval_results = task_results
            pipeline.broadcast_status("eval", "done")
            pipeline.broadcast({
                "type": "eval_complete",
                "data": {"results": task_results},
            })
        except Exception as e:
            pipeline.broadcast_status("eval", "error", error=str(e))
            traceback.print_exc()

    thread = threading.Thread(target=run, daemon=True)
    pipeline.active_thread = thread
    thread.start()
    return {"status": "started"}


@router.get("/results")
async def eval_results() -> dict:
    return {
        "status": pipeline.stage_status["eval"],
        "results": pipeline.eval_results,
    }
