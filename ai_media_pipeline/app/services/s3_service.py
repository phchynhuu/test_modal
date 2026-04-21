"""
S3 Service — all AWS S3 / CloudFront interactions live here.

Responsibilities:
  • Generate pre-signed PUT URLs so clients upload directly (bypasses backend).
  • Generate pre-signed GET URLs (or CloudFront-signed URLs) for results.
  • Build canonical S3 keys for input / output / temp objects.
  • Provide a thin async wrapper; boto3 calls are run in a thread pool so they
    don't block the event loop.
"""

import uuid
import mimetypes
import asyncio
from functools import partial
from typing import Literal

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

settings = get_settings()

# One boto3 client per process — boto3 is thread-safe for clients created once.
_s3_client = boto3.client(
    "s3",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
)


# ── Key helpers ───────────────────────────────────────────────────────────────

Bucket = Literal["input", "output", "temp"]

_PREFIX_MAP: dict[Bucket, str] = {
    "input":  settings.s3_input_prefix,
    "output": settings.s3_output_prefix,
    "temp":   settings.s3_temp_prefix,
}


def build_s3_key(
    bucket_area: Bucket,
    task_id: str,
    filename: str,
) -> str:
    """
    Deterministic key layout:  media/{area}/{task_id}/{filename}
    Keeping task_id as a path segment makes per-task listing trivial and
    gives CloudFront a natural invalidation pattern.
    """
    prefix = _PREFIX_MAP[bucket_area]
    return f"{prefix}/{task_id}/{filename}"


def _unique_filename(original: str) -> str:
    """Replace the base name with a UUID while preserving the extension."""
    ext = original.rsplit(".", 1)[-1] if "." in original else ""
    safe_ext = f".{ext}" if ext else ""
    return f"{uuid.uuid4().hex}{safe_ext}"


# ── Async wrappers ────────────────────────────────────────────────────────────

async def _run_in_executor(fn, *args, **kwargs):
    """Run a blocking boto3 call in the default thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


async def generate_presigned_upload_url(
    original_filename: str,
    content_type: str,
    task_id: str | None = None,
) -> dict[str, str | int]:
    """
    Returns a pre-signed S3 PUT URL.

    The client:
      1. PUTs the file directly to this URL (no backend hop).
      2. Passes the returned `s3_key` as the *_url field in processing requests.

    We use a temp prefix so S3 lifecycle policies auto-delete unprocessed
    uploads after 24 hours without any lambda / cron clean-up.
    """
    tid = task_id or uuid.uuid4().hex
    filename = _unique_filename(original_filename)
    key = build_s3_key("temp", tid, filename)

    try:
        url = await _run_in_executor(
            _s3_client.generate_presigned_url,
            "put_object",
            Params={
                "Bucket": settings.s3_bucket,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=settings.s3_presigned_url_expiry,
        )
    except ClientError as exc:
        raise RuntimeError(f"Failed to generate pre-signed URL: {exc}") from exc

    return {
        "upload_url": url,
        "s3_key": key,
        "expires_in": settings.s3_presigned_url_expiry,
        "task_id": tid,
    }


async def generate_presigned_download_url(s3_key: str) -> str:
    """
    Returns a time-limited URL for reading a result object.

    If a CloudFront domain is configured we return a plain HTTPS URL using
    the CDN (no signed cookies needed for public distributions).  This is the
    preferred path in production because CloudFront caches aggressively.
    """
    if settings.cloudfront_domain:
        # CloudFront serves the file; no per-request signing needed for
        # public distributions.  For private distributions swap this out
        # for CloudFront signed URL generation.
        return f"https://{settings.cloudfront_domain}/{s3_key}"

    try:
        url = await _run_in_executor(
            _s3_client.generate_presigned_url,
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": s3_key},
            ExpiresIn=settings.s3_result_url_expiry,
        )
    except ClientError as exc:
        raise RuntimeError(f"Failed to generate download URL: {exc}") from exc

    return url


async def upload_bytes(
    data: bytes,
    s3_key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload in-memory bytes (e.g. AI-generated image) and return the key."""
    try:
        await _run_in_executor(
            _s3_client.put_object,
            Bucket=settings.s3_bucket,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )
    except ClientError as exc:
        raise RuntimeError(f"S3 upload failed: {exc}") from exc
    return s3_key


async def object_exists(s3_key: str) -> bool:
    """Fast HEAD check — used for deduplication cache validation."""
    try:
        await _run_in_executor(
            _s3_client.head_object,
            Bucket=settings.s3_bucket,
            Key=s3_key,
        )
        return True
    except ClientError:
        return False
