"""
Cache Service — Redis-backed store for:
  1. Task state & progress  (TTL = 1 h after completion)
  2. Deduplication index    (TTL = 24 h; keyed by content hash)

Using redis-py async client so all operations are non-blocking inside the
FastAPI event loop and inside Celery async tasks.
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis, from_url

from app.config import get_settings

settings = get_settings()

# ── Module-level client — one connection pool per process ─────────────────────
_redis: Redis | None = None


def get_redis() -> Redis:
    """
    Lazy singleton.  Call once during startup (via lifespan) or rely on
    first-access initialisation.  Both patterns are safe because connection
    pools are thread-safe.
    """
    global _redis
    if _redis is None:
        _redis = from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


# ── Key builders ──────────────────────────────────────────────────────────────

def _task_key(task_id: str) -> str:
    return f"task:{task_id}"


def _dedup_key(content_hash: str) -> str:
    return f"dedup:{content_hash}"


# ── Task state ────────────────────────────────────────────────────────────────

async def set_task_state(task_id: str, state: dict[str, Any]) -> None:
    """Persist the full task state dict, serialised as JSON."""
    redis = get_redis()
    payload = json.dumps(state, default=str)   # `default=str` handles datetime
    await redis.setex(_task_key(task_id), settings.progress_ttl_seconds, payload)


async def get_task_state(task_id: str) -> dict[str, Any] | None:
    redis = get_redis()
    raw = await redis.get(_task_key(task_id))
    if raw is None:
        return None
    return json.loads(raw)


async def update_task_progress(task_id: str, progress: int) -> None:
    """
    Partial update — only mutates the `progress` field.
    Uses a GET-then-SET; acceptable because progress updates are idempotent
    and the only writer per task_id is the assigned Celery worker.
    """
    state = await get_task_state(task_id) or {}
    state["progress"] = max(0, min(100, progress))
    await set_task_state(task_id, state)


async def mark_task_completed(
    task_id: str,
    result_url: str,
    operation_type: str,
) -> None:
    state = await get_task_state(task_id) or {}
    state.update(
        {
            "status": "completed",
            "progress": 100,
            "result_url": result_url,
            "operation_type": operation_type,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    await set_task_state(task_id, state)


async def mark_task_failed(task_id: str, error: str) -> None:
    state = await get_task_state(task_id) or {}
    state.update(
        {
            "status": "failed",
            "error": error,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    await set_task_state(task_id, state)


# ── Deduplication ─────────────────────────────────────────────────────────────

def compute_request_hash(params: dict[str, Any]) -> str:
    """
    Stable, order-independent SHA-256 of the task parameters.

    The hash is intentionally derived from *logical* inputs (source URLs +
    operation params) rather than raw bytes so that two requests for the same
    logical operation return the same hash even if submitted from different
    clients.
    """
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def get_cached_result(content_hash: str) -> str | None:
    """Returns the S3 result URL if this exact task was already completed."""
    redis = get_redis()
    return await redis.get(_dedup_key(content_hash))


async def cache_result(content_hash: str, result_url: str) -> None:
    """Store the result URL keyed by content hash for 24 h."""
    redis = get_redis()
    await redis.setex(
        _dedup_key(content_hash),
        settings.cache_ttl_seconds,
        result_url,
    )
