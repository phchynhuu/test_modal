"""
Operation Router — factory pattern for dispatching tasks to the right handler.

Why a router instead of if/else chains?
  • Adding a new operation type means registering one entry in _REGISTRY — no
    touching of conditional logic spread across multiple call-sites.
  • The registry is inspectable at runtime (useful for /health or /capabilities
    endpoints that expose supported operations).
  • Each handler is a typed dataclass so IDEs can autocomplete its fields and
    static analysis catches missing arguments.

Architecture:
  OperationDescriptor  — static metadata about one operation type
  OperationRouter      — resolves an incoming request to an OperationDescriptor
                          and the correct Celery task + queue
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from app.models.requests import (
    MediaType,
    ImageOperation,
    VideoOperation,
)


# ── Operation descriptor ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class OperationDescriptor:
    """
    All metadata needed to dispatch one type of media operation.

    Fields:
      name          — human-readable label (also used as cache-key prefix)
      queue         — Celery queue to route the task onto
      task_name     — dotted Celery task name (registered in workers/tasks.py)
      build_payload — callable(request_dict) → dict of kwargs for the Celery task
      is_heavy      — True for GPU-bound tasks; surfaced in progress estimates
    """
    name: str
    queue: str
    task_name: str
    build_payload: Callable[[dict[str, Any]], dict[str, Any]]
    is_heavy: bool = False


# ── Registry ──────────────────────────────────────────────────────────────────
# Each entry maps a (operation_key) → OperationDescriptor.
# The operation_key is constructed by the router from the incoming request.

_REGISTRY: dict[str, OperationDescriptor] = {

    # ── Text → Image ──────────────────────────────────────────────────────
    "generate:image": OperationDescriptor(
        name="text_to_image",
        queue="media_generate_queue",
        task_name="app.workers.tasks.generate_media_task",
        build_payload=lambda r: {
            "operation": "generate_image",
            "prompt": r["prompt"],
            "style": r.get("style", ""),
            "width": r.get("width", 1024),
            "height": r.get("height", 1024),
            "provider": r.get("image_provider", "stability"),
            "webhook_url": r.get("webhook_url"),
        },
        is_heavy=False,
    ),

    # ── Text → Video ──────────────────────────────────────────────────────
    "generate:video": OperationDescriptor(
        name="text_to_video",
        queue="video_processing_queue",
        task_name="app.workers.tasks.generate_media_task",
        build_payload=lambda r: {
            "operation": "generate_video",
            "prompt": r["prompt"],
            "style": r.get("style", ""),
            "duration": r.get("duration", 5),
            "provider": r.get("video_provider", "runway"),
            "webhook_url": r.get("webhook_url"),
        },
        is_heavy=True,
    ),

    # ── Image → Video (Animate) ───────────────────────────────────────────
    "image_to_video": OperationDescriptor(
        name="image_to_video",
        queue="video_processing_queue",
        task_name="app.workers.tasks.image_to_video_task",
        build_payload=lambda r: {
            "image_url": str(r["image_url"]),
            "prompt": r.get("prompt", ""),
            "motion_scale": r.get("motion_scale", 0.5),
            "duration": r.get("duration", 4),
            "webhook_url": r.get("webhook_url"),
        },
        is_heavy=True,
    ),

    # ── Image → Image: edit ───────────────────────────────────────────────
    f"edit_image:{ImageOperation.EDIT}": OperationDescriptor(
        name="edit_image",
        queue="image_edit_queue",
        task_name="app.workers.tasks.edit_image_task",
        build_payload=lambda r: {
            "image_url": str(r["image_url"]),
            "prompt": r["prompt"],
            "mask_url": str(r["mask_url"]) if r.get("mask_url") else None,
            "operation": ImageOperation.EDIT,
            "webhook_url": r.get("webhook_url"),
        },
    ),

    # ── Image → Image: inpaint ────────────────────────────────────────────
    f"edit_image:{ImageOperation.INPAINT}": OperationDescriptor(
        name="inpaint_image",
        queue="image_edit_queue",
        task_name="app.workers.tasks.edit_image_task",
        build_payload=lambda r: {
            "image_url": str(r["image_url"]),
            "prompt": r["prompt"],
            "mask_url": str(r["mask_url"]),
            "operation": ImageOperation.INPAINT,
            "webhook_url": r.get("webhook_url"),
        },
    ),

    # ── Image → Image: style transfer ─────────────────────────────────────
    f"edit_image:{ImageOperation.STYLE}": OperationDescriptor(
        name="style_image",
        queue="image_edit_queue",
        task_name="app.workers.tasks.edit_image_task",
        build_payload=lambda r: {
            "image_url": str(r["image_url"]),
            "prompt": r["prompt"],
            "mask_url": None,
            "operation": ImageOperation.STYLE,
            "webhook_url": r.get("webhook_url"),
        },
    ),

    # ── Image → Image: upscale ────────────────────────────────────────────
    f"edit_image:{ImageOperation.UPSCALE}": OperationDescriptor(
        name="upscale_image",
        queue="image_edit_queue",
        task_name="app.workers.tasks.edit_image_task",
        build_payload=lambda r: {
            "image_url": str(r["image_url"]),
            "prompt": r.get("prompt", ""),
            "mask_url": None,
            "operation": ImageOperation.UPSCALE,
            "webhook_url": r.get("webhook_url"),
        },
    ),

    # ── Video → Video: trim ───────────────────────────────────────────────
    f"edit_video:{VideoOperation.TRIM}": OperationDescriptor(
        name="trim_video",
        queue="video_processing_queue",
        task_name="app.workers.tasks.edit_video_task",
        build_payload=lambda r: {
            "video_url": str(r["video_url"]),
            "operation": VideoOperation.TRIM,
            "start": r.get("start", 0),
            "end": r.get("end", 60),
            "face_source_url": None,
            "webhook_url": r.get("webhook_url"),
        },
        is_heavy=True,
    ),

    # ── Video → Video: style transfer ─────────────────────────────────────
    f"edit_video:{VideoOperation.STYLE}": OperationDescriptor(
        name="style_video",
        queue="video_processing_queue",
        task_name="app.workers.tasks.edit_video_task",
        build_payload=lambda r: {
            "video_url": str(r["video_url"]),
            "operation": VideoOperation.STYLE,
            "start": r.get("start", 0),
            "end": r.get("end", 60),
            "face_source_url": None,
            "webhook_url": r.get("webhook_url"),
        },
        is_heavy=True,
    ),

    # ── Video → Video: face swap ──────────────────────────────────────────
    f"edit_video:{VideoOperation.FACE_SWAP}": OperationDescriptor(
        name="face_swap_video",
        queue="video_processing_queue",
        task_name="app.workers.tasks.edit_video_task",
        build_payload=lambda r: {
            "video_url": str(r["video_url"]),
            "operation": VideoOperation.FACE_SWAP,
            "start": r.get("start", 0),
            "end": r.get("end", 60),
            "face_source_url": str(r["face_source_url"]) if r.get("face_source_url") else None,
            "webhook_url": r.get("webhook_url"),
        },
        is_heavy=True,
    ),
}


# ── Router ────────────────────────────────────────────────────────────────────

class OperationRouter:
    """
    Resolves incoming request data to the correct OperationDescriptor.

    Usage:
        router = OperationRouter()
        descriptor = router.resolve("edit_image", operation="inpaint")
        payload = descriptor.build_payload(request.model_dump())
    """

    def resolve(self, route_type: str, **kwargs) -> OperationDescriptor:
        """
        Build the registry key and look up the descriptor.

        route_type examples: "generate", "image_to_video", "edit_image", "edit_video"
        kwargs carries disambiguating fields (e.g. type="image", operation="inpaint")
        """
        key = self._build_key(route_type, **kwargs)
        descriptor = _REGISTRY.get(key)
        if descriptor is None:
            available = list(_REGISTRY.keys())
            raise ValueError(
                f"No handler registered for key '{key}'. "
                f"Available: {available}"
            )
        return descriptor

    @staticmethod
    def _build_key(route_type: str, **kwargs) -> str:
        if route_type == "generate":
            media_type = kwargs.get("type") or kwargs.get("media_type", "")
            return f"generate:{media_type}"

        if route_type == "image_to_video":
            return "image_to_video"

        if route_type in ("edit_image", "edit_video"):
            operation = kwargs.get("operation", "")
            return f"{route_type}:{operation}"

        return route_type

    @staticmethod
    def all_operations() -> list[str]:
        """Expose all registered keys — handy for a /capabilities endpoint."""
        return list(_REGISTRY.keys())


# Module-level singleton — import and use directly
router = OperationRouter()
