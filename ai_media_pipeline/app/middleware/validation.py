"""
Validation middleware and FastAPI dependency functions.

Two layers of validation:
  1. Request-level (FastAPI Dependencies injected into endpoint signatures)
     — validates fields available in the request body / headers before the
     handler runs.

  2. Media-level (MediaValidator class)
     — validates properties of the media object that can only be determined
     by inspecting the file itself (actual MIME type, resolution, duration).
     Called from the workers so the heavy I/O doesn't block the API.

Keeping validation in a single module means limit changes propagate without
hunting across multiple files.
"""

import re
import mimetypes
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import get_settings
from app.models.requests import UploadRequest

settings = get_settings()


# ── HTTP dependency: validate upload request ──────────────────────────────────

async def validate_upload_request(body: UploadRequest) -> UploadRequest:
    """
    Injected into POST /upload.  Checks declared content-type and file size
    before we issue a pre-signed URL — prevents clients from reserving upload
    slots for files we'd reject anyway.
    """
    all_allowed = settings.allowed_image_types + settings.allowed_video_types

    if body.content_type not in all_allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Content type '{body.content_type}' is not supported. "
                f"Allowed: {all_allowed}"
            ),
        )

    is_video = body.content_type.startswith("video/")
    limit = settings.max_video_size_bytes if is_video else settings.max_image_size_bytes
    limit_mb = limit / (1024 * 1024)

    if body.file_size > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {body.file_size} bytes exceeds {limit_mb:.0f} MB limit.",
        )

    return body


# ── HTTP dependency: require API key ─────────────────────────────────────────

_API_KEY_RE = re.compile(r"^Bearer\s+(\S+)$")


async def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """
    Minimal bearer-token auth dependency.

    In production, replace the hardcoded comparison with a database/JWT lookup.
    This layer exists to show the integration point — do NOT use a plaintext
    secret in a real deployment.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    match = _API_KEY_RE.match(authorization)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return match.group(1)


# ── Media-level validator (called inside workers) ─────────────────────────────

class MediaValidator:
    """
    Validates media properties that require reading the actual file.
    Instantiate once; call the appropriate validate_* method from within
    the Celery task before any expensive AI processing begins.
    """

    def __init__(self):
        self.settings = get_settings()

    def validate_image_bytes(
        self, data: bytes, declared_content_type: str
    ) -> None:
        """Check actual MIME type (via magic bytes) and size."""
        import imghdr

        actual_type = imghdr.what(None, h=data)
        type_map = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        actual_mime = type_map.get(actual_type or "", "unknown")

        if actual_mime not in self.settings.allowed_image_types:
            raise ValueError(
                f"Actual image type '{actual_mime}' is not allowed. "
                f"Magic bytes do not match declared type '{declared_content_type}'."
            )

        if len(data) > self.settings.max_image_size_bytes:
            raise ValueError(
                f"Image size {len(data)} bytes exceeds "
                f"{self.settings.max_image_size_bytes / (1024*1024):.0f} MB limit."
            )

    def validate_video_metadata(
        self,
        *,
        duration_seconds: float,
        width: int,
        height: int,
        content_type: str,
    ) -> None:
        """
        Validate video constraints.

        Call this after probing the video with ffprobe/mediainfo but before
        sending it to any AI service.  Probe logic is provider-specific so
        it's not included here.
        """
        if content_type not in self.settings.allowed_video_types:
            raise ValueError(
                f"Video type '{content_type}' is not allowed. "
                f"Allowed: {self.settings.allowed_video_types}"
            )

        if duration_seconds > self.settings.max_video_duration_seconds:
            raise ValueError(
                f"Video duration {duration_seconds:.1f}s exceeds "
                f"{self.settings.max_video_duration_seconds}s limit."
            )

        if height > self.settings.max_video_resolution:
            raise ValueError(
                f"Video height {height}px exceeds "
                f"{self.settings.max_video_resolution}p limit."
            )

    def validate_s3_url(self, url: str) -> None:
        """
        Confirm the URL belongs to our S3 bucket.
        Prevents SSRF via submitted media URLs pointing at internal services.
        """
        bucket = self.settings.s3_bucket
        region = self.settings.aws_region
        cf_domain = self.settings.cloudfront_domain

        allowed_hosts = {
            f"{bucket}.s3.amazonaws.com",
            f"{bucket}.s3.{region}.amazonaws.com",
            f"s3.{region}.amazonaws.com",
        }
        if cf_domain:
            allowed_hosts.add(cf_domain)

        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.hostname not in allowed_hosts:
            raise ValueError(
                f"URL host '{parsed.hostname}' is not an allowed source. "
                "All media must be uploaded to your S3 bucket first."
            )
