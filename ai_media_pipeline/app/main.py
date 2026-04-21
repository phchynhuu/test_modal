"""
FastAPI application entry point.

Lifespan context manager handles startup / shutdown:
  • Validates that Redis is reachable before accepting traffic.
  • Closes the Redis connection pool cleanly on shutdown to prevent
    "connection was closed in the middle of a command" errors in tests.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers.media import media_router
from app.services.cache_service import get_redis

settings = get_settings()
logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info("Starting %s…", settings.app_name)
    redis = get_redis()
    await redis.ping()   # raises if Redis is unreachable → pod fails fast
    logger.info("Redis connection verified.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    await redis.aclose()
    logger.info("Redis connection closed.")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Scalable async AI media processing pipeline. "
        "Generate, edit, and animate images and videos using state-of-the-art AI models."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Tighten `allow_origins` in production to your exact frontend domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(media_router, prefix=settings.api_v1_prefix)


# ── Health / meta endpoints ───────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health_check():
    """Used by load-balancer probes and k8s liveness checks."""
    return {"status": "ok"}


@app.get("/v1/capabilities", tags=["Meta"])
async def capabilities():
    """Returns all registered operation keys — useful for client introspection."""
    from app.services.operation_router import router as op_router
    return {"operations": op_router.all_operations()}
