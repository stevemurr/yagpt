"""SQLite persistence for experiments and runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

_db: aiosqlite.Connection | None = None

DB_PATH = Path("./data/experiments.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    data_config TEXT NOT NULL DEFAULT '{}',
    active_modules TEXT NOT NULL DEFAULT '["pretrain"]',
    module_configs TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    module_id TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    config_snapshot TEXT NOT NULL DEFAULT '{}',
    metrics_summary TEXT NOT NULL DEFAULT '{}',
    checkpoint_path TEXT,
    eval_results TEXT
);
"""


async def init_db() -> None:
    """Initialize the database connection and create tables."""
    global _db
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _db = await aiosqlite.connect(str(DB_PATH))
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=ON")
    await _db.executescript(SCHEMA)
    await _db.commit()


async def close_db() -> None:
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None


def _get_db() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("data_config", "active_modules", "module_configs", "config_snapshot", "metrics_summary", "eval_results"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


# --- Experiments CRUD ---


async def create_experiment(name: str, data_config: dict | None = None, active_modules: list[str] | None = None, module_configs: dict | None = None) -> dict[str, Any]:
    db = _get_db()
    now = _now()
    cursor = await db.execute(
        "INSERT INTO experiments (name, created_at, updated_at, data_config, active_modules, module_configs) VALUES (?, ?, ?, ?, ?, ?)",
        (name, now, now, json.dumps(data_config or {}), json.dumps(active_modules or ["pretrain"]), json.dumps(module_configs or {})),
    )
    await db.commit()
    return await get_experiment(cursor.lastrowid)  # type: ignore[arg-type]


async def list_experiments() -> list[dict[str, Any]]:
    db = _get_db()
    cursor = await db.execute("SELECT * FROM experiments ORDER BY updated_at DESC")
    rows = await cursor.fetchall()
    return [_row_to_dict(row) for row in rows]


async def get_experiment(experiment_id: int) -> dict[str, Any]:
    db = _get_db()
    cursor = await db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = await cursor.fetchone()
    if not row:
        raise ValueError(f"Experiment {experiment_id} not found")
    return _row_to_dict(row)


async def update_experiment(experiment_id: int, **kwargs: Any) -> dict[str, Any]:
    db = _get_db()
    sets = ["updated_at = ?"]
    vals: list[Any] = [_now()]
    for key in ("name", "data_config", "active_modules", "module_configs"):
        if key in kwargs:
            sets.append(f"{key} = ?")
            val = kwargs[key]
            vals.append(json.dumps(val) if isinstance(val, (dict, list)) else val)
    vals.append(experiment_id)
    await db.execute(f"UPDATE experiments SET {', '.join(sets)} WHERE id = ?", vals)
    await db.commit()
    return await get_experiment(experiment_id)


async def delete_experiment(experiment_id: int) -> None:
    db = _get_db()
    await db.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
    await db.commit()


# --- Runs ---


async def create_run(experiment_id: int, module_id: str, config_snapshot: dict | None = None) -> dict[str, Any]:
    db = _get_db()
    now = _now()
    cursor = await db.execute(
        "INSERT INTO runs (experiment_id, module_id, started_at, status, config_snapshot) VALUES (?, ?, ?, 'running', ?)",
        (experiment_id, module_id, now, json.dumps(config_snapshot or {})),
    )
    await db.commit()
    row_id = cursor.lastrowid
    cursor2 = await db.execute("SELECT * FROM runs WHERE id = ?", (row_id,))
    row = await cursor2.fetchone()
    return _row_to_dict(row)


async def list_runs(experiment_id: int) -> list[dict[str, Any]]:
    db = _get_db()
    cursor = await db.execute(
        "SELECT * FROM runs WHERE experiment_id = ? ORDER BY started_at DESC",
        (experiment_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_dict(row) for row in rows]


async def complete_run(
    run_id: int,
    status: str,
    metrics_summary: dict | None = None,
    checkpoint_path: str | None = None,
    eval_results: dict | None = None,
) -> None:
    db = _get_db()
    now = _now()
    await db.execute(
        "UPDATE runs SET completed_at = ?, status = ?, metrics_summary = ?, checkpoint_path = ?, eval_results = ? WHERE id = ?",
        (now, status, json.dumps(metrics_summary or {}), checkpoint_path, json.dumps(eval_results) if eval_results else None, run_id),
    )
    await db.commit()
