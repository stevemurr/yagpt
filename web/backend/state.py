"""Pipeline state singleton — holds model, status, metrics, and WebSocket connections."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from fastapi import WebSocket


@dataclass
class PipelineState:
    model: Any | None = None  # GPT instance
    tokenizer: Any | None = None  # Tokenizer instance
    checkpoint_path: str | None = None
    checkpoint_config: dict | None = None

    stage_status: dict[str, str] = field(default_factory=lambda: {
        "data": "idle", "pretrain": "idle", "sft": "idle",
        "lora": "idle", "align": "idle", "eval": "idle", "generate": "idle",
    })
    metrics: dict[str, deque] = field(default_factory=lambda: {
        "pretrain": deque(maxlen=10000),
        "sft": deque(maxlen=10000),
        "align": deque(maxlen=10000),
    })
    eval_results: dict | None = None

    ws_connections: set[WebSocket] = field(default_factory=set)
    stop_event: threading.Event = field(default_factory=threading.Event)
    active_thread: threading.Thread | None = None
    loop: asyncio.AbstractEventLoop | None = None

    # Experiment tracking
    current_experiment_id: int | None = None
    current_run_id: int | None = None

    def is_busy(self) -> bool:
        return self.active_thread is not None and self.active_thread.is_alive()

    def broadcast(self, msg: dict) -> None:
        """Thread-safe broadcast to all WebSocket connections."""
        if not self.loop or not self.ws_connections:
            return
        data = json.dumps(msg)
        for ws in list(self.ws_connections):
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(data), self.loop)
            except Exception:
                self.ws_connections.discard(ws)

    def broadcast_status(self, stage: str, status: str, **extra: Any) -> None:
        self.stage_status[stage] = status
        msg: dict[str, Any] = {"type": "status_update", "data": {"stage": stage, "status": status}}
        msg["data"].update(extra)
        self.broadcast(msg)

    def run_async(self, coro: Any) -> Any:
        """Run an async coroutine from a sync thread context."""
        if not self.loop:
            return None
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=10)

    def load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint and set up model + tokenizer."""
        from yagpt.models.gpt import GPT, GPTConfig
        from yagpt.tokenizer import Tokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]

        tokenizer = Tokenizer(cfg.get("tokenizer", "gpt2"))
        model_cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            n_kv_heads=cfg.get("n_kv_heads"),
            dim=cfg["dim"],
            hidden_dim=cfg.get("hidden_dim"),
            max_seq_len=cfg.get("max_seq_len", 2048),
            qk_norm=cfg.get("qk_norm", True),
            pad_vocab_to=cfg.get("pad_vocab_to", 64),
        )
        model = GPT(model_cfg)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_path = path
        self.checkpoint_config = cfg
        return ckpt

    def clear_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.tokenizer = None
        self.checkpoint_path = None
        self.checkpoint_config = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global singleton
pipeline = PipelineState()
