"""
Central configuration — all settings pulled from environment variables.
Uses pydantic-settings so every value is validated at startup; a misconfigured
deployment fails loudly instead of silently misbehaving at runtime.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ────────────────────────────────────────────────────────────────
    app_name: str = "AI Media Pipeline"
    api_v1_prefix: str = "/v1"
    debug: bool = False

    # ── AWS / S3 ───────────────────────────────────────────────────────────
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"
    s3_bucket: str
    cloudfront_domain: str = ""            # optional; falls back to S3 URL
    s3_presigned_url_expiry: int = 900     # seconds (15 min) for upload URLs
    s3_result_url_expiry: int = 86400      # seconds (24 h) for result URLs

    # S3 key prefixes — lifecycle policy should auto-delete /temp/ after 24 h
    s3_input_prefix: str = "media/input"
    s3_output_prefix: str = "media/output"
    s3_temp_prefix: str = "media/temp"

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400         # 24 h dedup cache lifetime
    progress_ttl_seconds: int = 3600       # 1 h progress key lifetime

    # ── Celery ────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Validation limits ─────────────────────────────────────────────────
    max_image_size_bytes: int = 50 * 1024 * 1024   # 50 MB
    max_video_size_bytes: int = 500 * 1024 * 1024  # 500 MB
    max_video_duration_seconds: int = 60
    max_video_resolution: int = 1080       # height in pixels
    allowed_image_types: list[str] = ["image/jpeg", "image/png", "image/webp"]
    allowed_video_types: list[str] = ["video/mp4", "video/quicktime"]

    # ── AI provider API keys ───────────────────────────────────────────────
    stability_api_key: str = ""            # Stability AI (image gen / edit)
    runway_api_key: str = ""               # Runway (video gen / image-to-video)
    replicate_api_token: str = ""          # Replicate (fallback / open models)
    gemini_api_key: str = ""               # Google Gemini (Imagen 3 / Veo 2)
    openai_api_key: str = ""               # OpenAI (DALL-E 3 / Sora)

    # ── Worker retry policy ───────────────────────────────────────────────
    task_max_retries: int = 3
    task_retry_backoff_base: int = 60      # seconds; doubles each attempt


@lru_cache
def get_settings() -> Settings:
    """Singleton — imported everywhere as `from app.config import get_settings`."""
    return Settings()
