"""Environment-driven application settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_VERSION = "1.0.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ECOVISION_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "EcoVision AI"
    app_env: str = "development"
    model_path: Path | None = None
    model_dir: Path = Path("ml/models")
    evaluation_dir: Path = Path("ml/evaluation")
    device: str = "cpu"
    data_dir: Path = Path("data")
    db_path: Path | None = None
    upload_dir: Path | None = None
    max_upload_mb: int = 8
    max_concurrent_inference: int = 2
    inference_timeout_seconds: float = 20.0
    # 0 selects the machine's core count; 1 is fastest for single-image CPU
    # inference and avoids oversubscription on fractional-CPU instances.
    torch_threads: int = 1
    # Grad-CAM runs on demand, outside the classification request.
    gradcam_enabled: bool = True
    gradcam_timeout_seconds: float = 30.0
    serve_frontend: bool = True
    frontend_dir: Path = Path("frontend")
    allowed_origins: str = "*"
    log_level: str = "INFO"

    # External model delivery (Hugging Face Hub). Empty repo disables downloads.
    hf_repo: str = ""
    hf_filename: str = "mobilenet_v2_final.pt"
    hf_manifest_filename: str = "model_manifest.json"
    hf_metrics_filename: str = "metrics.json"
    hf_token: str | None = None
    fetch_model: bool = True

    @field_validator(
        "model_path",
        "model_dir",
        "evaluation_dir",
        "data_dir",
        "db_path",
        "upload_dir",
        "frontend_dir",
        mode="after",
    )
    @classmethod
    def _resolve(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _derive_storage_paths(self) -> "Settings":
        if self.db_path is None:
            self.db_path = self.data_dir / "ecovision.db"
        if self.upload_dir is None:
            self.upload_dir = self.data_dir / "uploads"
        return self

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
