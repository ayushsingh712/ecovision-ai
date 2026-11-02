"""Image classification endpoint."""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.api.deps import (
    AdviceServiceDep,
    DatabaseDep,
    ModelServiceDep,
    SemaphoreDep,
    SettingsDep,
)
from backend.core.config import Settings
from backend.core.constants import LOW_CONFIDENCE_THRESHOLD, MEDIA_PREFIX
from backend.core.errors import AppError
from backend.core.logging import get_logger
from backend.database.database import Database
from backend.database.repositories import insert_prediction
from backend.schemas.prediction import ExplanationResponse, PredictionResponse
from backend.services.advice_service import AdviceService
from backend.services.explanation_service import build_explanation
from backend.services.image_service import (
    HEATMAP_FILE,
    ORIGINAL_FILE,
    OVERLAY_FILE,
    delete_prediction_images,
    save_original_image,
    validate_image_bytes,
)
from backend.services.model_service import ModelService
from ml.utils import utc_now_iso

router = APIRouter(tags=["predictions"])
logger = get_logger("ecovision.predict")


def _classify(
    settings: Settings,
    database: Database,
    model_service: ModelService,
    advice_service: AdviceService,
    data: bytes,
    filename: str | None,
) -> PredictionResponse:
    started = time.perf_counter()
    validated = validate_image_bytes(data, filename, settings.max_upload_bytes)
    result = model_service.classify(validated.image)

    public_id = uuid4().hex
    created_at = utc_now_iso()
    try:
        original_file = save_original_image(
            settings.upload_dir, public_id, validated
        )
        insert_prediction(
            database,
            {
                "public_id": public_id,
                "original_filename": validated.filename,
                "mime_type": validated.mime_type,
                "size_bytes": validated.size_bytes,
                "sha256": validated.sha256,
                "width": validated.width,
                "height": validated.height,
                "predicted_class": result.predicted_class,
                "confidence": result.confidence,
                "top3": result.top3,
                "model_stage": result.stage,
                "model_version": result.version,
                "original_file": original_file,
                # Grad-CAM is generated on demand; fill these in only after
                # the explanation actually exists.
                "heatmap_file": "",
                "overlay_file": "",
                "created_at": created_at,
            },
        )
    except Exception:
        delete_prediction_images(settings.upload_dir, public_id)
        raise

    media = f"{MEDIA_PREFIX}/{public_id}"
    logger.info(
        "Classified %s as %s (%.3f) in %.0f ms [prediction_id=%s]",
        validated.filename,
        result.predicted_class,
        result.confidence,
        (time.perf_counter() - started) * 1000,
        public_id,
    )
    return PredictionResponse(
        prediction_id=public_id,
        filename=validated.filename,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        top3=result.top3,
        advice=advice_service.for_class(result.predicted_class),
        gradcam={
            "original_url": f"{media}/{ORIGINAL_FILE}",
            "heatmap_url": f"{media}/{HEATMAP_FILE}",
            "overlay_url": f"{media}/{OVERLAY_FILE}",
        },
        model_stage=result.stage,
        model_version=result.version,
        created_at=created_at,
        low_confidence=result.confidence < LOW_CONFIDENCE_THRESHOLD,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    settings: SettingsDep,
    database: DatabaseDep,
    model_service: ModelServiceDep,
    advice_service: AdviceServiceDep,
    semaphore: SemaphoreDep,
    image: UploadFile = File(...),
) -> PredictionResponse:
    if not model_service.available:
        raise AppError(
            503,
            "MODEL_UNAVAILABLE",
            "No model checkpoint is installed, so predictions are unavailable.",
            model_service.load_error,
        )

    filename = image.filename
    data = await image.read(settings.max_upload_bytes + 1)
    await image.close()

    try:
        async with semaphore:
            async with asyncio.timeout(settings.inference_timeout_seconds):
                return await run_in_threadpool(
                    _classify,
                    settings,
                    database,
                    model_service,
                    advice_service,
                    data,
                    filename,
                )
    except TimeoutError as exc:
        logger.warning(
            "Classification timed out after %.1fs [filename=%s]",
            settings.inference_timeout_seconds,
            filename or "upload",
        )
        raise AppError(
            504,
            "INFERENCE_TIMEOUT",
            "The analysis took too long. Please try again.",
        ) from exc


@router.post(
    "/predictions/{prediction_id}/explanation",
    response_model=ExplanationResponse,
)
async def explain_prediction(
    prediction_id: str,
    settings: SettingsDep,
    database: DatabaseDep,
    model_service: ModelServiceDep,
    semaphore: SemaphoreDep,
) -> ExplanationResponse:
    """Generate Grad-CAM on demand, outside the classification request."""
    try:
        async with semaphore:
            async with asyncio.timeout(settings.gradcam_timeout_seconds):
                return await run_in_threadpool(
                    build_explanation,
                    settings,
                    database,
                    model_service,
                    prediction_id,
                )
    except TimeoutError:
        logger.warning(
            "Explanation timed out after %.1fs [prediction_id=%s]",
            settings.gradcam_timeout_seconds,
            prediction_id,
        )
        return ExplanationResponse(
            prediction_id=prediction_id,
            status="failed",
            message=(
                "The explanation took too long to generate. The classification "
                "result is still valid."
            ),
        )
