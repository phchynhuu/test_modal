"""
Celery task definitions.

Design principles
─────────────────
1. Each task is a thin orchestration wrapper around MediaService.
   The actual AI calls live in services/media_service.py so they can be
   unit-tested without a running Celery worker.

2. All tasks are async-compatible via `asyncio.run()`.  Celery workers are
   sync by default; wrapping async service methods this way avoids requiring
   a separate async worker library.

3. Retry policy:
   • `autoretry_for` catches transient provider errors automatically.
   • Exponential backoff (base × 2^attempt) prevents thundering-herd.
   • After `max_retries` exhaustion Celery marks the task FAILURE and we
     write the error into Redis so the status endpoint returns it.

4. Progress updates: workers call `update_task_progress(task_id, pct)` at
   meaningful checkpoints.  Clients poll GET /v1/media/{task_id} to read it.

5. Webhook delivery: on completion, if the original request included a
   `webhook_url`, we fire a POST in a separate short-lived task so the
   main task isn't held open waiting for the client's server.
"""

import asyncio
import uuid
import logging
from datetime import datetime, timezone

import httpx
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.workers.celery_app import celery_app
from app.services.media_service import MediaService, AIProviderTransientError
from app.services.cache_service import (
    set_task_state,
    update_task_progress,
    mark_task_completed,
    mark_task_failed,
    cache_result,
)
from app.services.s3_service import generate_presigned_download_url
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Shared MediaService instance — re-used across task invocations in the same
# worker process (connection pool to AI providers is reused).
_media_service = MediaService()


# ── Base task class ───────────────────────────────────────────────────────────

class MediaTask(Task):
    """
    Custom base class that handles cross-cutting concerns:
      • Marks the task state as PROCESSING in Redis on_start.
      • Marks it FAILED in Redis on_failure (called even after retries exhausted).
    """

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        task_id_str = kwargs.get("task_id", task_id)
        asyncio.run(
            mark_task_failed(task_id_str, str(exc))
        )
        logger.error(
            "Task %s failed permanently: %s",
            task_id_str,
            exc,
            exc_info=einfo,
        )


# ── Shared task lifecycle helpers ─────────────────────────────────────────────

def _run(coro):
    """Convenience wrapper: run an async coroutine from a sync Celery task."""
    return asyncio.run(coro)


async def _initialise_task(task_id: str, operation_type: str) -> None:
    await set_task_state(
        task_id,
        {
            "task_id": task_id,
            "status": "processing",
            "progress": 0,
            "operation_type": operation_type,
            "result_url": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        },
    )


async def _finalise_task(
    task_id: str,
    output_s3_key: str,
    operation_type: str,
    content_hash: str | None,
    webhook_url: str | None,
) -> str:
    result_url = await generate_presigned_download_url(output_s3_key)
    await mark_task_completed(task_id, result_url, operation_type)

    if content_hash:
        # Store result in dedup cache so identical future requests get this URL
        await cache_result(content_hash, result_url)

    if webhook_url:
        # Fire webhook asynchronously — don't block task completion
        deliver_webhook.delay(
            webhook_url=webhook_url,
            task_id=task_id,
            status="completed",
            result_url=result_url,
        )

    return result_url


# ── Tasks ─────────────────────────────────────────────────────────────────────

@celery_app.task(
    base=MediaTask,
    bind=True,
    name="app.workers.tasks.generate_media_task",
    # Retry on transient provider errors; Celery raises Retry, not the original exc
    autoretry_for=(AIProviderTransientError,),
    max_retries=settings.task_max_retries,
    retry_backoff=settings.task_retry_backoff_base,
    retry_backoff_max=600,
    retry_jitter=True,
)
def generate_media_task(
    self: Task,
    *,
    task_id: str,
    operation: str,       # "generate_image" | "generate_video"
    prompt: str,
    style: str = "",
    width: int = 1024,
    height: int = 1024,
    duration: int = 5,
    provider: str = "",   # image/video provider; empty = use default per operation
    content_hash: str | None = None,
    webhook_url: str | None = None,
) -> str:
    """
    Handles both text-to-image and text-to-video generation.
    The `operation` field determines which MediaService method is called.
    """
    async def _run_task():
        await _initialise_task(task_id, operation)
        await update_task_progress(task_id, 5)

        try:
            if operation == "generate_image":
                await update_task_progress(task_id, 20)
                output_key = await _media_service.generate_image(
                    task_id=task_id,
                    prompt=prompt,
                    style=style,
                    width=width,
                    height=height,
                    provider=provider or "stability",
                )
            elif operation == "generate_video":
                await update_task_progress(task_id, 10)
                output_key = await _media_service.generate_video(
                    task_id=task_id,
                    prompt=prompt,
                    style=style,
                    duration=duration,
                    provider=provider or "runway",
                )
            else:
                raise ValueError(f"Unknown generate operation: {operation}")

            await update_task_progress(task_id, 90)
            return await _finalise_task(
                task_id, output_key, operation, content_hash, webhook_url
            )

        except SoftTimeLimitExceeded:
            await mark_task_failed(task_id, "Task exceeded time limit")
            raise

    return _run(asyncio.coroutine(_run_task)() if False else _run_task())


@celery_app.task(
    base=MediaTask,
    bind=True,
    name="app.workers.tasks.image_to_video_task",
    autoretry_for=(AIProviderTransientError,),
    max_retries=settings.task_max_retries,
    retry_backoff=settings.task_retry_backoff_base,
    retry_backoff_max=600,
    retry_jitter=True,
)
def image_to_video_task(
    self: Task,
    *,
    task_id: str,
    image_url: str,
    prompt: str = "",
    motion_scale: float = 0.5,
    duration: int = 4,
    content_hash: str | None = None,
    webhook_url: str | None = None,
) -> str:
    async def _run_task():
        await _initialise_task(task_id, "image_to_video")
        await update_task_progress(task_id, 5)

        try:
            await update_task_progress(task_id, 15)
            output_key = await _media_service.image_to_video(
                task_id=task_id,
                image_url=image_url,
                prompt=prompt,
                motion_scale=motion_scale,
                duration=duration,
            )
            await update_task_progress(task_id, 90)
            return await _finalise_task(
                task_id, output_key, "image_to_video", content_hash, webhook_url
            )

        except SoftTimeLimitExceeded:
            await mark_task_failed(task_id, "Task exceeded time limit")
            raise

    return _run(_run_task())


@celery_app.task(
    base=MediaTask,
    bind=True,
    name="app.workers.tasks.edit_image_task",
    autoretry_for=(AIProviderTransientError,),
    max_retries=settings.task_max_retries,
    retry_backoff=settings.task_retry_backoff_base,
    retry_backoff_max=600,
    retry_jitter=True,
)
def edit_image_task(
    self: Task,
    *,
    task_id: str,
    image_url: str,
    prompt: str,
    mask_url: str | None,
    operation: str,
    content_hash: str | None = None,
    webhook_url: str | None = None,
) -> str:
    async def _run_task():
        await _initialise_task(task_id, f"edit_image:{operation}")
        await update_task_progress(task_id, 10)

        try:
            output_key = await _media_service.edit_image(
                task_id=task_id,
                image_url=image_url,
                prompt=prompt,
                mask_url=mask_url,
                operation=operation,
            )
            await update_task_progress(task_id, 90)
            return await _finalise_task(
                task_id, output_key, f"edit_image:{operation}", content_hash, webhook_url
            )

        except SoftTimeLimitExceeded:
            await mark_task_failed(task_id, "Task exceeded time limit")
            raise

    return _run(_run_task())


@celery_app.task(
    base=MediaTask,
    bind=True,
    name="app.workers.tasks.edit_video_task",
    autoretry_for=(AIProviderTransientError,),
    max_retries=settings.task_max_retries,
    retry_backoff=settings.task_retry_backoff_base,
    retry_backoff_max=600,
    retry_jitter=True,
)
def edit_video_task(
    self: Task,
    *,
    task_id: str,
    video_url: str,
    operation: str,
    start: int = 0,
    end: int = 60,
    face_source_url: str | None = None,
    content_hash: str | None = None,
    webhook_url: str | None = None,
) -> str:
    async def _run_task():
        await _initialise_task(task_id, f"edit_video:{operation}")
        await update_task_progress(task_id, 5)

        try:
            # Report progress through predictable milestones
            await update_task_progress(task_id, 15)
            output_key = await _media_service.edit_video(
                task_id=task_id,
                video_url=video_url,
                operation=operation,
                start=start,
                end=end,
                face_source_url=face_source_url,
            )
            await update_task_progress(task_id, 90)
            return await _finalise_task(
                task_id, output_key, f"edit_video:{operation}", content_hash, webhook_url
            )

        except SoftTimeLimitExceeded:
            await mark_task_failed(task_id, "Task exceeded time limit")
            raise

    return _run(_run_task())


# ── Webhook delivery task ─────────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.tasks.deliver_webhook",
    autoretry_for=(httpx.RequestError, httpx.HTTPStatusError),
    max_retries=3,
    retry_backoff=30,
    retry_jitter=True,
    # Webhook delivery is low-priority; re-use the generate queue
    queue="media_generate_queue",
)
def deliver_webhook(
    *,
    webhook_url: str,
    task_id: str,
    status: str,
    result_url: str | None = None,
    error: str | None = None,
) -> None:
    """
    Fire-and-forget POST to the client's webhook endpoint.

    Payload schema:
      {
        "task_id": "...",
        "status": "completed" | "failed",
        "result_url": "https://...",   # present on success
        "error": "...",                # present on failure
      }

    We run this as its own task so:
      • It doesn't hold up the worker slot of the expensive GPU task.
      • It has its own retry budget (3 attempts) independent of the main task.
    """
    payload = {
        "task_id": task_id,
        "status": status,
        "result_url": result_url,
        "error": error,
    }
    # Sync HTTP call is fine here — this is a short-lived fire-and-forget task
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(webhook_url, json=payload)
        resp.raise_for_status()   # triggers autoretry on 5xx

    logger.info("Webhook delivered to %s for task %s", webhook_url, task_id)
