"""
Unit tests for the OperationRouter.

No I/O — pure logic tests that run in milliseconds.
"""

import pytest
from app.services.operation_router import OperationRouter, router
from app.models.requests import ImageOperation, VideoOperation


def test_resolve_generate_image():
    d = router.resolve("generate", type="image")
    assert d.queue == "media_generate_queue"
    assert d.task_name == "app.workers.tasks.generate_media_task"


def test_resolve_generate_video():
    d = router.resolve("generate", type="video")
    assert d.queue == "video_processing_queue"
    assert d.is_heavy is True


def test_resolve_image_to_video():
    d = router.resolve("image_to_video")
    assert d.queue == "video_processing_queue"


def test_resolve_edit_image_inpaint():
    d = router.resolve("edit_image", operation=ImageOperation.INPAINT)
    assert d.queue == "image_edit_queue"
    assert d.name == "inpaint_image"


def test_resolve_edit_video_face_swap():
    d = router.resolve("edit_video", operation=VideoOperation.FACE_SWAP)
    assert d.queue == "video_processing_queue"
    assert d.is_heavy is True


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="No handler registered"):
        router.resolve("generate", type="audio")


def test_all_operations_returns_list():
    ops = router.all_operations()
    assert isinstance(ops, list)
    assert len(ops) > 0
    assert "generate:image" in ops
    assert "image_to_video" in ops


def test_build_payload_generate_image():
    d = router.resolve("generate", type="image")
    payload = d.build_payload(
        {"prompt": "neon city", "style": "cyberpunk", "width": 512, "height": 512}
    )
    assert payload["operation"] == "generate_image"
    assert payload["prompt"] == "neon city"
    assert payload["style"] == "cyberpunk"


def test_build_payload_edit_video_trim():
    d = router.resolve("edit_video", operation=VideoOperation.TRIM)
    payload = d.build_payload(
        {"video_url": "https://s3.example.com/v.mp4", "start": 5, "end": 20}
    )
    assert payload["operation"] == VideoOperation.TRIM
    assert payload["start"] == 5
    assert payload["end"] == 20
