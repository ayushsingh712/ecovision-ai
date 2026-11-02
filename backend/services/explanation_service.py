"""Generate Grad-CAM outputs on demand, after classification has completed."""

from __future__ import annotations

import time

from PIL import Image

from backend.core.config import Settings
from backend.core.constants import MEDIA_PREFIX
from backend.core.errors import AppError
from backend.core.logging import get_logger
from backend.database.database import Database
from backend.database.repositories import (
    get_prediction_by_public_id,
    update_prediction_explanation,
)
from backend.schemas.prediction import ExplanationResponse, GradCAMUrls
from backend.services.image_service import (
    HEATMAP_FILE,
    ORIGINAL_FILE,
    OVERLAY_FILE,
    save_explanation_images,
)
from backend.services.model_service import ModelService

logger = get_logger("ecovision.explanation")


def _urls(public_id: str) -> GradCAMUrls:
    base = f"{MEDIA_PREFIX}/{public_id}"
    return GradCAMUrls(
        original_url=f"{base}/{ORIGINAL_FILE}",
        heatmap_url=f"{base}/{HEATMAP_FILE}",
        overlay_url=f"{base}/{OVERLAY_FILE}",
    )


def build_explanation(
    settings: Settings,
    database: Database,
    model_service: ModelService,
    public_id: str,
) -> ExplanationResponse:
    row = get_prediction_by_public_id(database, public_id)
    if row is None:
        raise AppError(404, "PREDICTION_NOT_FOUND", "That prediction does not exist.")

    urls = _urls(public_id)
    folder = settings.upload_dir / public_id
    heatmap_path = folder / HEATMAP_FILE
    overlay_path = folder / OVERLAY_FILE

    if heatmap_path.is_file() and overlay_path.is_file():
        return ExplanationResponse(
            prediction_id=public_id, status="ready", gradcam=urls
        )

    if not settings.gradcam_enabled:
        return ExplanationResponse(
            prediction_id=public_id,
            status="disabled",
            message="Grad-CAM explanations are disabled on this deployment.",
        )

    if not model_service.available:
        raise AppError(
            503,
            "MODEL_UNAVAILABLE",
            "No model checkpoint is installed, so explanations are unavailable.",
            model_service.load_error,
        )

    original_path = folder / ORIGINAL_FILE
    if not original_path.is_file():
        logger.warning(
            "Cannot explain prediction %s: stored image is missing.", public_id
        )
        return ExplanationResponse(
            prediction_id=public_id,
            status="failed",
            message=(
                "The stored image is no longer available, so an explanation "
                "cannot be generated."
            ),
        )

    loaded = model_service.loaded
    if loaded is None:
        raise AppError(
            503,
            "MODEL_UNAVAILABLE",
            "No model checkpoint is installed, so explanations are unavailable.",
        )
    try:
        class_index = list(loaded.class_names).index(row["predicted_class"])
    except ValueError:
        logger.error(
            "Prediction %s uses class %s, which is not in the active model order.",
            public_id,
            row["predicted_class"],
        )
        return ExplanationResponse(
            prediction_id=public_id,
            status="failed",
            message="The explanation could not be generated for this prediction.",
        )

    started = time.perf_counter()
    try:
        with Image.open(original_path) as source:
            display = source.convert("RGB")
        gradcam = model_service.explain(display, class_index)
        files = save_explanation_images(settings.upload_dir, public_id, gradcam)
        update_prediction_explanation(
            database,
            public_id,
            files["heatmap_file"],
            files["overlay_file"],
        )
    except Exception:
        logger.exception(
            "Grad-CAM generation failed for prediction %s.", public_id
        )
        return ExplanationResponse(
            prediction_id=public_id,
            status="failed",
            message=(
                "The explanation could not be generated. The classification "
                "result is still valid."
            ),
        )

    logger.info(
        "Explanation ready for prediction %s in %.0f ms.",
        public_id,
        (time.perf_counter() - started) * 1000,
    )
    return ExplanationResponse(
        prediction_id=public_id, status="ready", gradcam=urls
    )
