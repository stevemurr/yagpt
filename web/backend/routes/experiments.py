"""CRUD routes for experiments and their runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db

router = APIRouter()


# --- Pydantic models ---


class ExperimentCreate(BaseModel):
    name: str
    data_config: dict[str, Any] = {}
    active_modules: list[str] = ["pretrain"]
    module_configs: dict[str, Any] = {}


class ExperimentUpdate(BaseModel):
    name: str | None = None
    data_config: dict[str, Any] | None = None
    active_modules: list[str] | None = None
    module_configs: dict[str, Any] | None = None


# --- Endpoints ---


@router.get("/")
async def list_experiments() -> list[dict[str, Any]]:
    return await db.list_experiments()


@router.post("/")
async def create_experiment(req: ExperimentCreate) -> dict[str, Any]:
    return await db.create_experiment(
        name=req.name,
        data_config=req.data_config,
        active_modules=req.active_modules,
        module_configs=req.module_configs,
    )


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: int) -> dict[str, Any]:
    try:
        return await db.get_experiment(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Experiment not found")


@router.put("/{experiment_id}")
async def update_experiment(experiment_id: int, req: ExperimentUpdate) -> dict[str, Any]:
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        return await db.update_experiment(experiment_id, **updates)
    except ValueError:
        raise HTTPException(status_code=404, detail="Experiment not found")


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: int) -> dict[str, str]:
    try:
        await db.get_experiment(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Experiment not found")
    await db.delete_experiment(experiment_id)
    return {"status": "deleted"}


@router.get("/{experiment_id}/runs")
async def list_runs(experiment_id: int) -> list[dict[str, Any]]:
    try:
        await db.get_experiment(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return await db.list_runs(experiment_id)
