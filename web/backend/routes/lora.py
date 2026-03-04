"""POST /api/lora/apply, GET /api/lora/info — LoRA / QLoRA routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import pipeline

router = APIRouter()


class LoRAApplyRequest(BaseModel):
    checkpoint: str | None = None
    rank: int = 16
    alpha: int | None = None
    target_modules: list[str] | None = None
    dropout: float = 0.0
    qlora: bool = False
    block_size: int = 64


class LoRAInfoResponse(BaseModel):
    lora_params: int
    total_params: int
    ratio: float
    rank: int
    qlora: bool


@router.post("/apply", response_model=LoRAInfoResponse)
async def apply_lora_route(req: LoRAApplyRequest) -> LoRAInfoResponse:
    # Load checkpoint if needed
    if req.checkpoint and req.checkpoint != pipeline.checkpoint_path:
        try:
            pipeline.load_checkpoint(req.checkpoint)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}")

    if pipeline.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    from yagpt.lora.lora import apply_lora, count_lora_params

    # QLoRA: quantize first
    if req.qlora:
        from yagpt.lora.qlora import quantize_model_nf4
        quantize_model_nf4(pipeline.model, block_size=req.block_size)

    apply_lora(
        pipeline.model,
        rank=req.rank,
        alpha=req.alpha,
        target_modules=req.target_modules,
        dropout=req.dropout,
    )

    lora_p, total_p = count_lora_params(pipeline.model)
    pipeline.broadcast_status("lora", "done")

    return LoRAInfoResponse(
        lora_params=lora_p,
        total_params=total_p,
        ratio=round(lora_p / total_p * 100, 2) if total_p > 0 else 0,
        rank=req.rank,
        qlora=req.qlora,
    )


@router.get("/info")
async def lora_info() -> dict:
    if pipeline.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    from yagpt.lora.lora import count_lora_params
    lora_p, total_p = count_lora_params(pipeline.model)
    has_lora = lora_p > 0
    return {
        "has_lora": has_lora,
        "lora_params": lora_p,
        "total_params": total_p,
        "ratio": round(lora_p / total_p * 100, 2) if total_p > 0 and has_lora else 0,
    }
