"""FastAPI entry point for the YAGPT web backend."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .state import pipeline
from . import db
from .ws import router as ws_router
from .routes.generate import router as generate_router
from .routes.checkpoints import router as checkpoints_router
from .routes.pretrain import router as pretrain_router
from .routes.sft import router as sft_router
from .routes.lora import router as lora_router
from .routes.alignment import router as alignment_router
from .routes.eval import router as eval_router
from .routes.data import router as data_router
from .routes.experiments import router as experiments_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    pipeline.loop = asyncio.get_running_loop()
    await db.init_db()
    yield
    await db.close_db()
    pipeline.clear_model()


app = FastAPI(title="YAGPT Web", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)
app.include_router(generate_router, prefix="/api/generate", tags=["generate"])
app.include_router(checkpoints_router, prefix="/api/checkpoints", tags=["checkpoints"])
app.include_router(pretrain_router, prefix="/api/pretrain", tags=["pretrain"])
app.include_router(sft_router, prefix="/api/sft", tags=["sft"])
app.include_router(lora_router, prefix="/api/lora", tags=["lora"])
app.include_router(alignment_router, prefix="/api/alignment", tags=["alignment"])
app.include_router(eval_router, prefix="/api/eval", tags=["eval"])
app.include_router(data_router, prefix="/api/data", tags=["data"])
app.include_router(experiments_router, prefix="/api/experiments", tags=["experiments"])


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "has_model": pipeline.model is not None}


def run() -> None:
    uvicorn.run("web.backend.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
