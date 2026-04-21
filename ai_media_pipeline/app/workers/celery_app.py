"""
Celery application factory and queue configuration.

Queue separation rationale
──────────────────────────
  media_generate_queue   — text-to-image (standard GPU, moderate latency)
  image_edit_queue       — image edits / upscale (high priority, fast turnaround)
  video_processing_queue — everything video (heavy GPU, long-running)

Keeping queues separate means:
  • A surge of video jobs can't starve fast image edits.
  • We can scale worker groups independently (e.g. 2× video workers on
    high-VRAM nodes, 8× image workers on smaller nodes).
  • video_processing_queue tasks set a longer soft/hard time limit so they
    aren't killed mid-render.

Worker invocation examples:
  # Image-edit workers (lightweight, many instances)
  celery -A app.workers.celery_app worker -Q image_edit_queue --concurrency=8

  # Video workers (GPU-intensive, fewer instances)
  celery -A app.workers.celery_app worker -Q video_processing_queue --concurrency=2

  # Generate workers (standard GPU)
  celery -A app.workers.celery_app worker -Q media_generate_queue --concurrency=4
"""

from celery import Celery
from kombu import Queue, Exchange

from app.config import get_settings

settings = get_settings()

# ── App instance ──────────────────────────────────────────────────────────────

celery_app = Celery(
    "ai_media_pipeline",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],  # auto-discover tasks in this module
)

# ── Queue / Exchange definitions ──────────────────────────────────────────────

default_exchange = Exchange("media", type="direct")

celery_app.conf.task_queues = (
    Queue(
        "media_generate_queue",
        exchange=default_exchange,
        routing_key="media_generate",
        queue_arguments={"x-max-priority": 5},
    ),
    Queue(
        "image_edit_queue",
        exchange=default_exchange,
        routing_key="image_edit",
        # Higher priority ceiling → image edits jump ahead of pending generates
        queue_arguments={"x-max-priority": 10},
    ),
    Queue(
        "video_processing_queue",
        exchange=default_exchange,
        routing_key="video_processing",
        queue_arguments={"x-max-priority": 5},
    ),
)

# ── Task routing — maps task names to queues ──────────────────────────────────

celery_app.conf.task_routes = {
    # text-to-image / text-to-video share one task; queue is decided by the
    # operation_router at enqueue time and passed via `queue=` kwarg.
    "app.workers.tasks.generate_media_task":  {"queue": "media_generate_queue"},
    "app.workers.tasks.image_to_video_task":  {"queue": "video_processing_queue"},
    "app.workers.tasks.edit_image_task":      {"queue": "image_edit_queue"},
    "app.workers.tasks.edit_video_task":      {"queue": "video_processing_queue"},
}

# ── Global task settings ──────────────────────────────────────────────────────

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Acknowledgement — ack only after the task completes so crashes don't
    # silently drop work (requires idempotent tasks).
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Time limits
    # image tasks finish in seconds; video tasks may take several minutes
    task_soft_time_limit=600,    # 10 min — triggers SoftTimeLimitExceeded
    task_time_limit=660,         # 11 min — SIGKILL

    # Result retention — keep results in Redis for 24 h for status polling
    result_expires=86400,

    # Prevent duplicate execution due to broker redelivery
    task_acks_on_failure_or_timeout=False,

    # Beat schedule (optional — add periodic clean-up tasks here)
    beat_schedule={},
)
