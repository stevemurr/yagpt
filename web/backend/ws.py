"""WebSocket endpoint and broadcast helpers."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .state import pipeline

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    pipeline.ws_connections.add(ws)
    try:
        # Send current state on connect
        await ws.send_json({
            "type": "init",
            "data": {
                "stages": pipeline.stage_status,
                "checkpoint": pipeline.checkpoint_path,
                "has_model": pipeline.model is not None,
            },
        })
        # Keep alive — listen for pings or client messages
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        pipeline.ws_connections.discard(ws)
