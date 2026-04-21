"""
Unit tests for validation logic.
"""

import pytest
from pydantic import ValidationError

from app.models.requests import (
    GenerateMediaRequest,
    EditImageRequest,
    EditVideoRequest,
    ImageOperation,
    VideoOperation,
)
from app.middleware.validation import MediaValidator


# ── Request model validation ──────────────────────────────────────────────────

def test_generate_image_resolution_cap():
    with pytest.raises(ValidationError, match="Resolution capped"):
        GenerateMediaRequest(type="image", prompt="test", width=3840, height=2160)


def test_edit_image_inpaint_requires_mask():
    with pytest.raises(ValidationError, match="mask_url is required"):
        EditImageRequest(
            image_url="https://s3.example.com/img.png",
            prompt="fill it in",
            operation=ImageOperation.INPAINT,
        )


def test_edit_image_inpaint_with_mask_ok():
    req = EditImageRequest(
        image_url="https://s3.example.com/img.png",
        prompt="fill it in",
        mask_url="https://s3.example.com/mask.png",
        operation=ImageOperation.INPAINT,
    )
    assert req.operation == ImageOperation.INPAINT


def test_edit_video_end_before_start():
    with pytest.raises(ValidationError, match="end.*greater than.*start"):
        EditVideoRequest(
            video_url="https://s3.example.com/vid.mp4",
            operation=VideoOperation.TRIM,
            start=40,
            end=20,
        )


def test_edit_video_trim_window_too_long():
    with pytest.raises(ValidationError, match="cannot exceed 60 seconds"):
        EditVideoRequest(
            video_url="https://s3.example.com/vid.mp4",
            operation=VideoOperation.TRIM,
            start=0,
            end=90,
        )


def test_edit_video_face_swap_requires_source():
    with pytest.raises(ValidationError, match="face_source_url required"):
        EditVideoRequest(
            video_url="https://s3.example.com/vid.mp4",
            operation=VideoOperation.FACE_SWAP,
        )


# ── MediaValidator ────────────────────────────────────────────────────────────

def test_media_validator_rejects_long_video():
    v = MediaValidator()
    with pytest.raises(ValueError, match="duration"):
        v.validate_video_metadata(
            duration_seconds=120.0,
            width=1920,
            height=1080,
            content_type="video/mp4",
        )


def test_media_validator_rejects_high_res():
    v = MediaValidator()
    with pytest.raises(ValueError, match="height"):
        v.validate_video_metadata(
            duration_seconds=10.0,
            width=3840,
            height=2160,
            content_type="video/mp4",
        )


def test_media_validator_rejects_unsupported_type():
    v = MediaValidator()
    with pytest.raises(ValueError, match="not allowed"):
        v.validate_video_metadata(
            duration_seconds=10.0,
            width=1280,
            height=720,
            content_type="video/webm",
        )


def test_media_validator_valid_video():
    v = MediaValidator()
    # Should not raise
    v.validate_video_metadata(
        duration_seconds=30.0,
        width=1280,
        height=720,
        content_type="video/mp4",
    )


def test_media_validator_ssrf_protection():
    v = MediaValidator()
    with pytest.raises(ValueError, match="not an allowed source"):
        v.validate_s3_url("https://169.254.169.254/latest/meta-data/")
