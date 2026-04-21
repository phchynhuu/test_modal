"""
API-level integration tests.

Uses:
  • httpx.AsyncClient as the test client (avoids starting a real server)
  • fakeredis for an in-memory Redis (no external dependency)
  • unittest.mock to patch Celery task dispatch and S3 calls

Run with:  pytest tests/ -v
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import fakeredis.aioredis
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services import cache_service

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def fake_redis(monkeypatch):
    """Redirect all cache_service Redis calls to fakeredis."""
    server = fakeredis.FakeServer()
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(cache_service, "_redis", fake)
    yield fake
    await fake.aclose()


@pytest_asyncio.fixture
async def client():
    """Async test client wrapping the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as ac:
        yield ac


# ── Health check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Upload URL ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_url_image(client):
    mock_result = {
        "upload_url": "https://s3.example.com/presigned",
        "s3_key": "media/temp/abc123/image.png",
        "expires_in": 900,
        "task_id": "abc123",
    }
    with patch(
        "app.routers.media.generate_presigned_upload_url",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        resp = await client.post(
            "/v1/media/upload",
            json={"filename": "photo.png", "content_type": "image/png", "file_size": 1024},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "upload_url" in body
    assert "s3_key" in body


@pytest.mark.asyncio
async def test_upload_url_rejects_oversized(client):
    big = 100 * 1024 * 1024   # 100 MB — over 50 MB image limit
    resp = await client.post(
        "/v1/media/upload",
        json={"filename": "big.png", "content_type": "image/png", "file_size": big},
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_upload_url_rejects_unsupported_type(client):
    resp = await client.post(
        "/v1/media/upload",
        json={"filename": "doc.pdf", "content_type": "application/pdf", "file_size": 1024},
    )
    assert resp.status_code == 415


# ── Generate media ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_image_enqueues(client):
    with patch("app.routers.media.celery_app") as mock_celery:
        mock_celery.send_task = MagicMock()
        resp = await client.post(
            "/v1/media/generate",
            json={"type": "image", "prompt": "a sunset over mountains"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    assert body["queue"] == "media_generate_queue"
    assert "task_id" in body
    mock_celery.send_task.assert_called_once()


@pytest.mark.asyncio
async def test_generate_video_enqueues_to_video_queue(client):
    with patch("app.routers.media.celery_app") as mock_celery:
        mock_celery.send_task = MagicMock()
        resp = await client.post(
            "/v1/media/generate",
            json={"type": "video", "prompt": "a drone shot of a forest"},
        )
    assert resp.status_code == 202
    assert resp.json()["queue"] == "video_processing_queue"


# ── Deduplication ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_image_dedup_returns_cached(client, fake_redis):
    import json, hashlib
    payload = {"type": "image", "prompt": "a cat", "style": "", "webhook_url": None,
               "width": 1024, "height": 1024, "duration": 5}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    content_hash = hashlib.sha256(canonical.encode()).hexdigest()
    cached_url = "https://cdn.example.com/media/output/cat.png"
    await fake_redis.setex(f"dedup:{content_hash}", 3600, cached_url)

    with patch("app.routers.media.celery_app") as mock_celery:
        mock_celery.send_task = MagicMock()
        resp = await client.post("/v1/media/generate", json=payload)

    assert resp.status_code == 202
    assert resp.json()["status"] == "cached"
    mock_celery.send_task.assert_not_called()


# ── Image-to-video ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_image_to_video_enqueues(client):
    with patch("app.routers.media.celery_app") as mock_celery:
        mock_celery.send_task = MagicMock()
        resp = await client.post(
            "/v1/media/image-to-video",
            json={
                "image_url": "https://s3.example.com/media/input/img.png",
                "prompt": "gentle wind blowing",
                "motion_scale": 0.4,
                "duration": 4,
            },
        )
    assert resp.status_code == 202
    assert resp.json()["queue"] == "video_processing_queue"


# ── Edit image ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_image_inpaint_requires_mask(client):
    resp = await client.post(
        "/v1/media/edit-image",
        json={
            "image_url": "https://s3.example.com/img.png",
            "prompt": "remove the car",
            "operation": "inpaint",
            # mask_url intentionally omitted
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_edit_image_enqueues_to_image_queue(client):
    with patch("app.routers.media.celery_app") as mock_celery:
        mock_celery.send_task = MagicMock()
        resp = await client.post(
            "/v1/media/edit-image",
            json={
                "image_url": "https://s3.example.com/img.png",
                "prompt": "make it pop-art style",
                "operation": "style",
            },
        )
    assert resp.status_code == 202
    assert resp.json()["queue"] == "image_edit_queue"


# ── Edit video ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_video_trim_invalid_window(client):
    resp = await client.post(
        "/v1/media/edit-video",
        json={
            "video_url": "https://s3.example.com/vid.mp4",
            "operation": "trim",
            "start": 50,
            "end": 30,   # end < start → invalid
        },
    )
    assert resp.status_code == 422


# ── Task status ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_task_status_not_found(client):
    resp = await client.get("/v1/media/nonexistent-task-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_task_status_returns_state(client, fake_redis):
    task_id = "abc123test"
    state = {
        "task_id": task_id,
        "status": "processing",
        "progress": 45,
        "operation_type": "generate_image",
        "result_url": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    import json
    await fake_redis.setex(f"task:{task_id}", 3600, json.dumps(state))

    resp = await client.get(f"/v1/media/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    assert body["progress"] == 45


# ── Auth ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_auth_rejected():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        # No Authorization header
    ) as ac:
        resp = await ac.get("/v1/media/some-task-id")
    assert resp.status_code == 401
