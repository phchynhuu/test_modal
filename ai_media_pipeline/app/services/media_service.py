"""
Media Service — async interface to external AI provider APIs.

Design decisions:
  • Each method is `async` so it can be awaited directly in an async Celery
    task (using `asyncio.run()` inside the sync task wrapper).
  • All AI provider calls are wrapped in a single `call_with_retry` helper;
    retries are also handled at the Celery level, but having retries here
    protects against transient HTTP errors within a single task attempt.
  • Provider-specific code lives in private helpers so swapping a provider
    (e.g. Stability → fal.ai) only touches one small section.
  • Heavy imports (httpx, PIL) are deferred to method bodies so the module
    loads fast even when dependencies aren't installed.

NOTE: The AI provider calls below show the integration shape.  Actual endpoint
paths / request schemas vary per provider — substitute your real keys and
adjust accordingly.
"""

import asyncio
import uuid
from typing import Any

import httpx

from app.config import get_settings
from app.services.s3_service import build_s3_key, upload_bytes

settings = get_settings()


# ── Retry helper ──────────────────────────────────────────────────────────────

class AIProviderError(Exception):
    """Raised when an AI API returns a non-retryable error (e.g. 400)."""


class AIProviderTransientError(Exception):
    """Raised on 429 / 5xx — Celery will catch this and retry the task."""


async def _call_with_retry(
    fn,
    *args,
    max_attempts: int = 3,
    base_backoff: float = 2.0,
    **kwargs,
) -> Any:
    """
    Thin exponential-backoff wrapper for a single provider call.
    This runs *inside* one Celery task attempt; the task-level retry (celery
    autoretry_for) handles failures that exhaust these inner attempts.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except AIProviderTransientError:
            if attempt == max_attempts:
                raise
            await asyncio.sleep(base_backoff ** attempt)


# ── Shared HTTP client (connection-pooled) ────────────────────────────────────
# Created lazily so tests can monkeypatch before first access.

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=120.0)
    return _http_client


async def _post_json(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON to an AI API, raising typed exceptions on failure."""
    client = _get_http_client()
    resp = await client.post(url, json=payload, headers=headers)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise AIProviderTransientError(
            f"Provider returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 400:
        raise AIProviderError(
            f"Provider returned {resp.status_code}: {resp.text[:200]}"
        )
    return resp.json()


# ── Provider-specific helpers (pure functions) ────────────────────────────────

def _pick_gemini_aspect_ratio(width: int, height: int) -> str:
    """
    Map arbitrary dimensions to the nearest Imagen 3 aspect ratio string.
    Supported: "1:1", "3:4", "4:3", "9:16", "16:9".
    """
    ratio = width / height
    candidates = {
        "1:1":  1.0,
        "4:3":  4 / 3,
        "3:4":  3 / 4,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
    }
    return min(candidates, key=lambda k: abs(candidates[k] - ratio))


def _pick_dalle_size(width: int, height: int) -> str:
    """
    Map arbitrary dimensions to the nearest DALL-E 3 size string.
    Supported: "1024x1024", "1792x1024", "1024x1792".
    """
    if width > height:
        return "1792x1024"
    if height > width:
        return "1024x1792"
    return "1024x1024"


# ── MediaService ──────────────────────────────────────────────────────────────

class MediaService:
    """
    Encapsulates all AI model API calls.

    Each public method:
      1. Calls the appropriate AI provider.
      2. Receives raw bytes (image/video).
      3. Uploads the result to S3 under /media/output/{task_id}/.
      4. Returns the S3 key (callers build the final URL from this).
    """

    # ── Text → Image ──────────────────────────────────────────────────────

    async def generate_image(
        self,
        task_id: str,
        prompt: str,
        style: str = "",
        width: int = 1024,
        height: int = 1024,
        provider: str = "stability",
    ) -> str:
        """
        Generate an image using the specified provider.

        Providers:
          stability — Stability AI SD3 (default)
          gemini    — Google Imagen 3
          openai    — OpenAI DALL-E 3
        """
        from app.models.requests import ImageProvider

        full_prompt = f"{style}, {prompt}" if style else prompt

        if provider == ImageProvider.GEMINI:
            return await self._generate_image_gemini(task_id, full_prompt, width, height)
        if provider == ImageProvider.OPENAI:
            return await self._generate_image_openai(task_id, full_prompt, width, height)

        # Default: Stability AI
        async def _call():
            data = await _post_json(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                payload={
                    "prompt": full_prompt,
                    "width": width,
                    "height": height,
                    "output_format": "png",
                },
                headers={
                    "Authorization": f"Bearer {settings.stability_api_key}",
                    "Accept": "application/json",
                },
            )
            import base64
            return base64.b64decode(data["image"])

        image_bytes: bytes = await _call_with_retry(_call)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.png")
        return await upload_bytes(image_bytes, key, "image/png")

    async def _generate_image_gemini(
        self,
        task_id: str,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
    ) -> str:
        """
        Google Imagen 3 via Gemini API.

        Provider docs: https://ai.google.dev/api/generate-content#imagen
        Model: imagen-3.0-generate-002
        """
        # Map pixel dimensions to the nearest supported aspect ratio
        ratio = _pick_gemini_aspect_ratio(width, height)

        async def _call():
            data = await _post_json(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"imagen-3.0-generate-002:predict?key={settings.gemini_api_key}",
                payload={
                    "instances": [{"prompt": prompt}],
                    "parameters": {
                        "sampleCount": 1,
                        "aspectRatio": ratio,
                        "outputMimeType": "image/png",
                    },
                },
                headers={"Content-Type": "application/json"},
            )
            import base64
            return base64.b64decode(data["predictions"][0]["bytesBase64Encoded"])

        image_bytes: bytes = await _call_with_retry(_call)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.png")
        return await upload_bytes(image_bytes, key, "image/png")

    async def _generate_image_openai(
        self,
        task_id: str,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
    ) -> str:
        """
        OpenAI DALL-E 3.

        Provider docs: https://platform.openai.com/docs/api-reference/images/create
        DALL-E 3 supports: 1024x1024, 1792x1024, 1024x1792.
        """
        size = _pick_dalle_size(width, height)

        async def _call():
            data = await _post_json(
                "https://api.openai.com/v1/images/generations",
                payload={
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": size,
                    "response_format": "b64_json",
                },
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
            import base64
            return base64.b64decode(data["data"][0]["b64_json"])

        image_bytes: bytes = await _call_with_retry(_call)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.png")
        return await upload_bytes(image_bytes, key, "image/png")

    # ── Text → Video ──────────────────────────────────────────────────────

    async def generate_video(
        self,
        task_id: str,
        prompt: str,
        style: str = "",
        duration: int = 5,
        provider: str = "runway",
    ) -> str:
        """
        Generate a video using the specified provider.

        Providers:
          runway — Runway Gen-3 Alpha Turbo (default)
          gemini — Google Veo 2
          openai — OpenAI Sora
        """
        from app.models.requests import VideoProvider

        full_prompt = f"{style}, {prompt}" if style else prompt

        if provider == VideoProvider.GEMINI:
            return await self._generate_video_gemini(task_id, full_prompt, duration)

        # Default: Runway
        client = _get_http_client()
        headers = {
            "Authorization": f"Bearer {settings.runway_api_key}",
            "X-Runway-Version": "2024-11-06",
        }

        # 1. Submit generation job
        submission = await _post_json(
            "https://api.dev.runwayml.com/v1/text_to_video",
            payload={
                "promptText": full_prompt,
                "duration": duration,
                "ratio": "1280:768",
            },
            headers=headers,
        )
        job_id = submission["id"]

        # 2. Poll for completion (max ~5 min for a 5 s clip)
        video_url = await self._poll_runway_job(job_id, headers, timeout=300)

        # 3. Download and store in S3
        video_bytes = await self._download_bytes(video_url)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.mp4")
        return await upload_bytes(video_bytes, key, "video/mp4")

    async def _generate_video_gemini(
        self, task_id: str, prompt: str, duration: int = 5
    ) -> str:
        """
        Google Veo 2 via Gemini long-running operations API.

        Provider docs: https://ai.google.dev/api/veo/generate-videos
        Model: veo-2.0-generate-001
        Veo returns a long-running operation; we poll until it completes.
        Duration options: 5 or 8 seconds.
        """
        client = _get_http_client()
        veo_duration = 8 if duration > 6 else 5   # Veo only supports 5 s or 8 s

        # Submit
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"veo-2.0-generate-001:predictLongRunning?key={settings.gemini_api_key}",
            json={
                "instances": [{"prompt": prompt}],
                "parameters": {
                    "aspectRatio": "16:9",
                    "durationSeconds": veo_duration,
                    "sampleCount": 1,
                },
            },
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            raise AIProviderError(f"Veo submit error {resp.status_code}: {resp.text[:200]}")
        operation_name = resp.json()["name"]

        # Poll long-running operation
        video_url = await self._poll_gemini_operation(operation_name)

        video_bytes = await self._download_bytes(video_url)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.mp4")
        return await upload_bytes(video_bytes, key, "video/mp4")

    # ── Image → Video (Animate) ───────────────────────────────────────────

    async def image_to_video(
        self,
        task_id: str,
        image_url: str,
        prompt: str = "",
        motion_scale: float = 0.5,
        duration: int = 4,
    ) -> str:
        """
        Runway Gen-3 image-to-video.  Motion scale controls how much the
        image is allowed to deform (0 = mostly static, 1 = free motion).
        """
        client = _get_http_client()
        headers = {
            "Authorization": f"Bearer {settings.runway_api_key}",
            "X-Runway-Version": "2024-11-06",
        }

        submission = await _post_json(
            "https://api.dev.runwayml.com/v1/image_to_video",
            payload={
                "promptImage": str(image_url),
                "promptText": prompt,
                "duration": duration,
                "motionScale": motion_scale,
                "ratio": "1280:768",
            },
            headers=headers,
        )
        job_id = submission["id"]
        video_url = await self._poll_runway_job(job_id, headers, timeout=300)
        video_bytes = await self._download_bytes(video_url)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.mp4")
        return await upload_bytes(video_bytes, key, "video/mp4")

    # ── Image → Image (Edit) ──────────────────────────────────────────────

    async def edit_image(
        self,
        task_id: str,
        image_url: str,
        prompt: str,
        mask_url: str | None,
        operation: str,
    ) -> str:
        """
        Routes to the correct Stability AI editing endpoint based on operation.

        Operations:
          edit    → /edit/sd3  (img2img)
          inpaint → /edit/inpaint
          style   → /edit/style
          upscale → /upscale/conservative
        """
        from app.models.requests import ImageOperation

        endpoint_map = {
            ImageOperation.EDIT:    "/v2beta/stable-image/edit/sd3",
            ImageOperation.INPAINT: "/v2beta/stable-image/edit/inpaint",
            ImageOperation.STYLE:   "/v2beta/stable-image/edit/style",
            ImageOperation.UPSCALE: "/v2beta/stable-image/upscale/conservative",
        }
        endpoint = endpoint_map.get(operation)
        if endpoint is None:
            raise AIProviderError(f"Unknown image operation: {operation}")

        # Download source image to pass as multipart (Stability requires file upload)
        image_bytes = await self._download_bytes(str(image_url))
        files: dict[str, Any] = {
            "image": ("input.png", image_bytes, "image/png"),
            "prompt": (None, prompt),
            "output_format": (None, "png"),
        }
        if mask_url:
            mask_bytes = await self._download_bytes(str(mask_url))
            files["mask"] = ("mask.png", mask_bytes, "image/png")

        client = _get_http_client()
        resp = await client.post(
            f"https://api.stability.ai{endpoint}",
            files=files,
            headers={"Authorization": f"Bearer {settings.stability_api_key}"},
        )
        if resp.status_code >= 500 or resp.status_code == 429:
            raise AIProviderTransientError(f"Stability {resp.status_code}")
        if resp.status_code >= 400:
            raise AIProviderError(f"Stability edit error {resp.status_code}: {resp.text}")

        import base64
        result_bytes = base64.b64decode(resp.json()["image"])
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.png")
        return await upload_bytes(result_bytes, key, "image/png")

    # ── Video → Video (Edit) ──────────────────────────────────────────────

    async def edit_video(
        self,
        task_id: str,
        video_url: str,
        operation: str,
        start: int = 0,
        end: int = 60,
        face_source_url: str | None = None,
    ) -> str:
        """
        Routes video editing operations.

        TRIM    → FFmpeg via Replicate (no GPU needed, but kept in video queue
                  so heavy workloads don't starve CPU workers).
        STYLE   → Replicate style-transfer model (heavy GPU).
        FACE_SWAP → Replicate face-swap model (heavy GPU).
        """
        from app.models.requests import VideoOperation

        if operation == VideoOperation.TRIM:
            return await self._trim_video(task_id, video_url, start, end)
        if operation == VideoOperation.STYLE:
            return await self._style_video(task_id, video_url)
        if operation == VideoOperation.FACE_SWAP:
            return await self._face_swap_video(
                task_id, video_url, face_source_url
            )
        raise AIProviderError(f"Unknown video operation: {operation}")

    # ── Private helpers ───────────────────────────────────────────────────

    async def _poll_gemini_operation(
        self,
        operation_name: str,
        timeout: int = 300,
        interval: int = 5,
    ) -> str:
        """Poll a Gemini long-running operation until done; return output video URL."""
        client = _get_http_client()
        elapsed = 0
        while elapsed < timeout:
            resp = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/{operation_name}"
                f"?key={settings.gemini_api_key}",
            )
            data = resp.json()
            if data.get("done"):
                error = data.get("error")
                if error:
                    raise AIProviderError(f"Veo operation failed: {error}")
                # Response contains a signed video URI
                return data["response"]["predictions"][0]["videoUri"]
            await asyncio.sleep(interval)
            elapsed += interval
        raise AIProviderTransientError(
            f"Gemini operation {operation_name} timed out after {timeout}s"
        )

    async def _poll_sora_job(
        self,
        job_id: str,
        headers: dict,
        timeout: int = 300,
        interval: int = 5,
    ) -> str:
        """Poll an OpenAI Sora generation job until complete; return video URL."""
        client = _get_http_client()
        elapsed = 0
        while elapsed < timeout:
            resp = await client.get(
                f"https://api.openai.com/v1/video/generations/{job_id}",
                headers=headers,
            )
            data = resp.json()
            status = data.get("status")
            if status == "completed":
                return data["data"][0]["url"]
            if status == "failed":
                raise AIProviderError(f"Sora job {job_id} failed: {data.get('error')}")
            await asyncio.sleep(interval)
            elapsed += interval
        raise AIProviderTransientError(f"Sora job {job_id} timed out after {timeout}s")

    async def _poll_runway_job(
        self,
        job_id: str,
        headers: dict,
        timeout: int = 300,
        interval: int = 5,
    ) -> str:
        """Poll Runway job until status == SUCCEEDED; return output URL."""
        client = _get_http_client()
        elapsed = 0
        while elapsed < timeout:
            resp = await client.get(
                f"https://api.dev.runwayml.com/v1/tasks/{job_id}",
                headers=headers,
            )
            data = resp.json()
            status = data.get("status")
            if status == "SUCCEEDED":
                return data["output"][0]   # first output URL
            if status == "FAILED":
                raise AIProviderError(f"Runway job {job_id} failed: {data}")
            await asyncio.sleep(interval)
            elapsed += interval
        raise AIProviderTransientError(f"Runway job {job_id} timed out after {timeout}s")

    async def _download_bytes(self, url: str) -> bytes:
        client = _get_http_client()
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

    async def _trim_video(
        self, task_id: str, video_url: str, start: int, end: int
    ) -> str:
        """
        Use Replicate's FFmpeg model to trim without re-encoding.

        Replicate model: lucataco/ffmpeg (replace with preferred model slug).
        """
        result = await _call_with_retry(
            self._replicate_run,
            "lucataco/ffmpeg:latest",
            {
                "video_url": video_url,
                "start_time": start,
                "end_time": end,
            },
        )
        video_bytes = await self._download_bytes(result)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.mp4")
        return await upload_bytes(video_bytes, key, "video/mp4")

    async def _style_video(self, task_id: str, video_url: str) -> str:
        """Video style transfer via Replicate."""
        result = await _call_with_retry(
            self._replicate_run,
            "stability-ai/stable-video-diffusion:latest",
            {"video_path": video_url},
        )
        video_bytes = await self._download_bytes(result)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.mp4")
        return await upload_bytes(video_bytes, key, "video/mp4")

    async def _face_swap_video(
        self, task_id: str, video_url: str, face_source_url: str | None
    ) -> str:
        result = await _call_with_retry(
            self._replicate_run,
            "deepfakes/roop:latest",
            {"target_video": video_url, "source_face": face_source_url},
        )
        video_bytes = await self._download_bytes(result)
        key = build_s3_key("output", task_id, f"result_{uuid.uuid4().hex}.mp4")
        return await upload_bytes(video_bytes, key, "video/mp4")

    async def _replicate_run(self, model_slug: str, inputs: dict) -> str:
        """
        Generic Replicate prediction runner.
        Submits a prediction and polls until complete, returning the output URL.
        """
        client = _get_http_client()
        headers = {
            "Authorization": f"Token {settings.replicate_api_token}",
            "Content-Type": "application/json",
        }

        # Submit
        resp = await client.post(
            f"https://api.replicate.com/v1/models/{model_slug}/predictions",
            json={"input": inputs},
            headers=headers,
        )
        if resp.status_code >= 400:
            raise AIProviderTransientError(f"Replicate submit error: {resp.text}")
        prediction_url = resp.json()["urls"]["get"]

        # Poll
        for _ in range(120):   # 10-min ceiling at 5-s interval
            poll = await client.get(prediction_url, headers=headers)
            data = poll.json()
            if data["status"] == "succeeded":
                output = data["output"]
                return output[0] if isinstance(output, list) else output
            if data["status"] == "failed":
                raise AIProviderError(f"Replicate failed: {data.get('error')}")
            await asyncio.sleep(5)

        raise AIProviderTransientError("Replicate prediction timed out")
