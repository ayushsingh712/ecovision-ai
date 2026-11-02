"""Install deployment model artifacts before the model service loads."""

from __future__ import annotations

from pathlib import Path

from backend.core.config import Settings
from backend.core.logging import get_logger
from ml.remote import download_model_bundle

logger = get_logger("ecovision.artifacts")


def fetch_model_artifacts(settings: Settings) -> Path | None:
    """Download the configured Hugging Face bundle, if any.

    Failures are logged and swallowed: the API must still boot in degraded mode
    so that ``/api/health`` and the frontend banner remain available.
    """
    if settings.model_path is not None:
        logger.info(
            "ECOVISION_MODEL_PATH is set; skipping remote artifact download."
        )
        return None
    repo = settings.hf_repo.strip()
    if not settings.fetch_model or not repo:
        return None

    logger.info("Downloading model artifacts from Hugging Face repo %s", repo)
    try:
        return download_model_bundle(
            repo_id=repo,
            filename=settings.hf_filename,
            models_dir=settings.model_dir,
            evaluation_dir=settings.evaluation_dir,
            manifest_filename=settings.hf_manifest_filename,
            metrics_filename=settings.hf_metrics_filename,
            token=settings.hf_token or None,
        )
    except Exception:
        logger.exception(
            "Model artifact download failed; the API will start in degraded mode."
        )
        return None
