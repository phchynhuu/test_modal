"""
FastAPI router — all /v1/media/* endpoints.

Each endpoint follows the same pattern:
  1. Parse + validate the request body (Pydantic handles this automatically).
  2. Compute a content hash for deduplication; return cached result if hit.
  3. Resolve the operation descriptor via OperationRouter.
  4. Persist initial task state to Redis.
  5. Enqueue the Celery task on the correct queue.
  6. Return a TaskResponse immediately (202 Accepted) — processing is async.

The client then polls GET /v1/media/{task_id} for status updates.
"""

import uuid
from datetime import datetime, timezone

from celery import current_app as celery_current_app
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.middleware.validation import require_api_key
from app.models.requests import (
    EditImageRequest,
    EditVideoRequest,
    GenerateMediaRequest,
    ImageToVideoRequest,
    UploadRequest,
)
from app.models.responses import (
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
    UploadURLResponse,
)
from app.services.cache_service import (
    compute_request_hash,
    get_cached_result,
    get_task_state,
    set_task_state,
)
from app.services.operation_router import router as op_router
from app.services.s3_service import generate_presigned_upload_url
from app.workers.celery_app import celery_app

settings = get_settings()

media_router = APIRouter(
    prefix="/media",
    tags=["Media"],
    dependencies=[Depends(require_api_key)],   # auth on every route in this router
)

# ── Estimated wait times (seconds) — surfaced for UX purposes ────────────────
_QUEUE_ESTIMATES = {
    "media_generate_queue":   15,
    "image_edit_queue":        5,
    "video_processing_queue": 60,
}


def _make_task_id() -> str:
    return uuid.uuid4().hex


async def _check_dedup(content_hash: str) -> TaskResponse | None:
    """
    Returns a TaskResponse with status=CACHED if we already have a result
    for this exact input combination, otherwise returns None.
    """
    cached_url = await get_cached_result(content_hash)
    if not cached_url:
        return None

    task_id = _make_task_id()   # still give them a unique task_id for tracking
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.CACHED,
        queue="cache",
        estimated_wait_seconds=0,
        created_at=datetime.now(timezone.utc),
        # Attach result_url via extra field — callers can read it immediately
        # (We extend TaskResponse via model_extra in the final response)
    )


async def _enqueue(
    task_id: str,
    descriptor,           # OperationDescriptor
    payload: dict,
    content_hash: str,
) -> None:
    """
    Write initial Redis state and send the Celery task.

    Saving state *before* enqueueing means the status endpoint can answer
    immediately even before a worker picks up the task.
    """
    await set_task_state(
        task_id,
        {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "operation_type": descriptor.name,
            "result_url": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        },
    )

    # Inject task_id and content_hash into the payload
    payload["task_id"] = task_id
    payload["content_hash"] = content_hash

    # Send to the queue specified by the descriptor
    celery_app.send_task(
        descriptor.task_name,
        kwargs=payload,
        queue=descriptor.queue,
    )


# ── POST /v1/media/generate ───────────────────────────────────────────────────

@media_router.post(
    "/generate",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate image or video from a text prompt",
)
async def generate_media(body: GenerateMediaRequest) -> TaskResponse:
    content_hash = compute_request_hash(body.model_dump())

    cached = await _check_dedup(content_hash)
    if cached:
        return cached

    descriptor = op_router.resolve("generate", type=body.type)
    payload = descriptor.build_payload(body.model_dump())

    task_id = _make_task_id()
    await _enqueue(task_id, descriptor, payload, content_hash)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        queue=descriptor.queue,
        estimated_wait_seconds=_QUEUE_ESTIMATES.get(descriptor.queue),
        created_at=datetime.now(timezone.utc),
    )


# ── POST /v1/media/image-to-video ─────────────────────────────────────────────

@media_router.post(
    "/image-to-video",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Animate a still image into a short video clip",
)
async def image_to_video(body: ImageToVideoRequest) -> TaskResponse:
    content_hash = compute_request_hash(body.model_dump())

    cached = await _check_dedup(content_hash)
    if cached:
        return cached

    descriptor = op_router.resolve("image_to_video")
    payload = descriptor.build_payload(body.model_dump())

    task_id = _make_task_id()
    await _enqueue(task_id, descriptor, payload, content_hash)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        queue=descriptor.queue,
        estimated_wait_seconds=_QUEUE_ESTIMATES.get(descriptor.queue),
        created_at=datetime.now(timezone.utc),
    )


# ── POST /v1/media/edit-image ─────────────────────────────────────────────────

@media_router.post(
    "/edit-image",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Edit, inpaint, style-transfer or upscale an image",
)
async def edit_image(body: EditImageRequest) -> TaskResponse:
    content_hash = compute_request_hash(body.model_dump())

    cached = await _check_dedup(content_hash)
    if cached:
        return cached

    descriptor = op_router.resolve("edit_image", operation=body.operation)
    payload = descriptor.build_payload(body.model_dump())

    task_id = _make_task_id()
    await _enqueue(task_id, descriptor, payload, content_hash)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        queue=descriptor.queue,
        estimated_wait_seconds=_QUEUE_ESTIMATES.get(descriptor.queue),
        created_at=datetime.now(timezone.utc),
    )


# ── POST /v1/media/edit-video ─────────────────────────────────────────────────

@media_router.post(
    "/edit-video",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trim, style-transfer or face-swap a video",
)
async def edit_video(body: EditVideoRequest) -> TaskResponse:
    content_hash = compute_request_hash(body.model_dump())

    cached = await _check_dedup(content_hash)
    if cached:
        return cached

    descriptor = op_router.resolve("edit_video", operation=body.operation)
    payload = descriptor.build_payload(body.model_dump())

    task_id = _make_task_id()
    await _enqueue(task_id, descriptor, payload, content_hash)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        queue=descriptor.queue,
        estimated_wait_seconds=_QUEUE_ESTIMATES.get(descriptor.queue),
        created_at=datetime.now(timezone.utc),
    )


# ── POST /v1/media/upload ─────────────────────────────────────────────────────

@media_router.post(
    "/upload",
    response_model=UploadURLResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a pre-signed S3 URL for direct client upload",
    description="""
**Two-step upload flow (bypasses the backend entirely):**

1. POST here with filename, content_type, and file_size.
2. PUT your file binary directly to the returned `upload_url`
   (include `Content-Type` header matching what you declared here).
3. Pass the returned `s3_key` as the `image_url` / `video_url` field in any
   processing endpoint.  The pipeline reads the file directly from S3.

This pattern means large files never transit through the API server,
keeping p99 latency low and enabling unlimited horizontal API scaling.
    """,
)
async def get_upload_url(body: UploadRequest) -> UploadURLResponse:
    # validate_upload_request is called implicitly by Pydantic; explicit size
    # check here for the response max_size_bytes field
    is_video = body.content_type.startswith("video/")
    max_bytes = (
        settings.max_video_size_bytes if is_video else settings.max_image_size_bytes
    )

    if body.file_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Declared file_size {body.file_size} exceeds limit {max_bytes}.",
        )

    result = await generate_presigned_upload_url(
        original_filename=body.filename,
        content_type=body.content_type,
    )

    return UploadURLResponse(
        upload_url=result["upload_url"],
        s3_key=result["s3_key"],
        expires_in=settings.s3_presigned_url_expiry,
        max_size_bytes=max_bytes,
    )


# ── GET /v1/media/{task_id} ───────────────────────────────────────────────────

@media_router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll the status and result of a media task",
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    state = await get_task_state(task_id)

    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found. It may have expired.",
        )

    return TaskStatusResponse(
        task_id=state["task_id"],
        status=TaskStatus(state.get("status", "pending")),
        progress=state.get("progress", 0),
        result_url=state.get("result_url"),
        error=state.get("error"),
        created_at=datetime.fromisoformat(state["created_at"]),
        completed_at=(
            datetime.fromisoformat(state["completed_at"])
            if state.get("completed_at")
            else None
        ),
        operation_type=state.get("operation_type"),
    )
