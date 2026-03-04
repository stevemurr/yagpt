"""POST /api/generate — text generation from loaded model."""

from __future__ import annotations

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import pipeline

router = APIRouter()


class GenerateRequest(BaseModel):
    checkpoint: str | None = None
    prompt: str = "Once upon a time"
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = 50


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int


@router.post("", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    # Load checkpoint if specified and different from current
    if req.checkpoint and req.checkpoint != pipeline.checkpoint_path:
        try:
            pipeline.load_checkpoint(req.checkpoint)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}")

    if pipeline.model is None or pipeline.tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. Provide a checkpoint path.")

    device = next(pipeline.model.parameters()).device
    tokens = pipeline.tokenizer.encode(req.prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        output = pipeline.model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )

    generated = output[0].tolist()
    text = pipeline.tokenizer.decode(generated)
    new_tokens = len(generated) - len(tokens)
    return GenerateResponse(text=text, tokens_generated=new_tokens)
