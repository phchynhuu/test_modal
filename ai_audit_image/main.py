"""
Image Audit API — streaming + queue

POST /audit  →  text/event-stream (SSE)

Each check result is streamed as it completes.
A semaphore limits concurrent audits; excess requests wait in a queue
and receive their queue position before processing begins.
"""

import asyncio
import base64
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Literal

from openai import OpenAI
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from check_color import check_color
from check_photo_color import check_photo_color
from check_size import check_size
from check_texture import check_texture_scale

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_CONCURRENT = 3    # audits processed simultaneously
MAX_QUEUE      = 50   # requests allowed to wait before 503

# OpenAI-compatible settings — override via environment variables
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "gpt-4o")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # None → default api.openai.com

PROMPT_DIR = Path(__file__).parent / "prompt"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

ImageType = Literal["3d", "illustration", "photography"]

# AI checks per image type: list of (event_name, prompt_file)
AI_CHECKS: dict[str, list[tuple[str, str]]] = {
    "3d": [
        ("style",  "3d/audit_style.txt"),
        ("layout", "3d/audit_layout.txt"),
        ("crop",   "audit_crop.txt"),
    ],
    "illustration": [
        ("layout", "illustration/audit_layout.txt"),
        ("stroke", "illustration/audit_stroke.txt"),
        ("crop",   "audit_crop.txt"),
        ("texture", "illustration/audit_texture.txt"),
    ],
    "photography": [
        ("safezone", "photography/audit_safezone.txt"),
        ("text",     "photography/text_audit.txt"),
        ("crop",     "audit_crop.txt"),
        ("expression", "photography/audit_expression.txt"),
        ("layout", "photography/audit_layout.txt"),
        ("pose", "photography/audit_pose.txt"),
    ],
}

# ---------------------------------------------------------------------------
# App lifecycle — create semaphore & thread pool inside the event loop
# ---------------------------------------------------------------------------

_semaphore: asyncio.Semaphore
_executor:  ThreadPoolExecutor
_queue_depth: int = 0   # approximate count of requests waiting for the semaphore


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _semaphore, _executor
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    _executor  = ThreadPoolExecutor(max_workers=MAX_CONCURRENT * 3, thread_name_prefix="audit")
    yield
    _executor.shutdown(wait=False)


app = FastAPI(
    title="Image Audit API",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_prompt(relative_path: str) -> str:
    return (PROMPT_DIR / relative_path).read_text(encoding="utf-8")


def _get_media_type(filename: str) -> str:
    return MEDIA_TYPE_MAP.get(Path(filename).suffix.lower(), "image/png")


def _extract_json(text: str) -> dict:
    """Strip optional markdown fences then parse JSON."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    return json.loads(text.strip())


# Sync — runs inside the thread pool so it won't block the event loop
def _ai_check_sync(prompt: str, image_path: str, media_type: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64}"

    client = OpenAI(base_url=OPENAI_BASE_URL)  # api_key from OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _extract_json(response.choices[0].message.content)


async def _run(fn, *args):
    """Run a blocking function in the shared thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


def _sse(event: str, data: dict) -> str:
    """Format a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _is_fail(result: dict) -> bool:
    return str(result.get("result", "")).lower() == "fail"


# ---------------------------------------------------------------------------
# Streaming audit generator
# ---------------------------------------------------------------------------

async def _audit_stream(
    tmp_path: str,
    image_type: str,
    media_type: str,
) -> AsyncGenerator[str, None]:
    global _queue_depth

    failed: list[str] = []

    # ── Place request in queue ─────────────────────────────────────────────
    _queue_depth += 1
    position = _queue_depth
    yield _sse("queued", {
        "position": position,
        "message": f"In queue — position {position}",
    })

    # Acquire semaphore (this is the actual wait point)
    try:
        await _semaphore.acquire()
    except asyncio.CancelledError:
        _queue_depth -= 1
        raise

    _queue_depth -= 1
    yield _sse("processing", {"message": "Audit started"})

    try:
        # ── 1. Size (all types) ────────────────────────────────────────────
        size_result = await _run(check_size, tmp_path)
        if not size_result["passed"]:
            failed.append("size")
        yield _sse("check", {"name": "size", "passed": size_result["passed"], "result": size_result})

        # ── 2. Programmatic color / texture checks ─────────────────────────
        if image_type == "illustration":
            color_result = await _run(check_color, tmp_path)
            if not color_result["is_valid"]:
                failed.append("color")
            yield _sse("check", {"name": "color", "passed": color_result["is_valid"], "result": color_result})

            texture_result = await _run(check_texture_scale, tmp_path)
            if not texture_result["is_valid"]:
                failed.append("texture")
            yield _sse("check", {"name": "texture", "passed": texture_result["is_valid"], "result": texture_result})

        elif image_type == "photography":
            color_result = await _run(check_photo_color, tmp_path)
            if not color_result["is_ok"]:
                failed.append("color")
            yield _sse("check", {"name": "color", "passed": color_result["is_ok"], "result": color_result})

        # ── 3. AI vision checks ────────────────────────────────────────────
        for check_name, prompt_path in AI_CHECKS[image_type]:
            prompt = _load_prompt(prompt_path)
            try:
                ai_result = await _run(_ai_check_sync, prompt, tmp_path, media_type)
                passed = not _is_fail(ai_result)
            except (json.JSONDecodeError, Exception) as exc:
                ai_result = {"result": "ERROR", "error": str(exc)}
                passed = False

            if not passed:
                failed.append(check_name)
            yield _sse("check", {"name": check_name, "passed": passed, "result": ai_result})

        # ── 4. Final summary ───────────────────────────────────────────────
        yield _sse("done", {
            "image_type":     image_type,
            "overall_result": "FAIL" if failed else "PASS",
            "failed_checks":  failed,
        })

    except Exception as exc:
        yield _sse("error", {"message": str(exc)})

    finally:
        _semaphore.release()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/audit",
    summary="Stream image audit results",
    response_description="Server-Sent Events stream",
)
async def audit_image(
    file: UploadFile = File(..., description="Image to audit (.jpg .jpeg .png .webp)"),
    image_type: ImageType = Form(..., description="3d | illustration | photography"),
) -> StreamingResponse:
    """
    Audit an image and **stream each check result** as a Server-Sent Event.

    **SSE event types**

    | Event | When | Key fields |
    |---|---|---|
    | `queued` | immediately | `position` |
    | `processing` | semaphore acquired | `message` |
    | `check` | after each check | `name`, `passed`, `result` |
    | `done` | all checks finished | `overall_result`, `failed_checks` |
    | `error` | unrecoverable failure | `message` |

    **Checks per image type**

    - **3D**: size · style · layout · crop
    - **Illustration**: size · color · texture · layout · stroke · crop
    - **Photography**: size · color · safezone · text · crop
    """
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{suffix}'. Accepted: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    if _queue_depth >= MAX_QUEUE:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy — {MAX_QUEUE} requests already queued. Try again later.",
        )

    content = await file.read()
    media_type = _get_media_type(file.filename or "")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    async def generate():
        try:
            async for chunk in _audit_stream(tmp_path, image_type, media_type):
                yield chunk
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
        },
    )


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    """Current queue depth and concurrency config."""
    return {
        "max_concurrent": MAX_CONCURRENT,
        "max_queue":      MAX_QUEUE,
        "queue_depth":    _queue_depth,
    }
