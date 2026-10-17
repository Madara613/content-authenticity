from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    environment: str = "development"
    cors_origins: str = "*"
    
    # Upload limits & constraints
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    allowed_mime_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    
    # Storage paths
    heatmap_dir: Path = Path("data/heatmaps")

    # Phase 2 Hugging Face Models
    ai_detector_model_id: str = "umm-maybe/AI-image-detector"
    clip_model_id: str = "openai/clip-vit-base-patch32"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
