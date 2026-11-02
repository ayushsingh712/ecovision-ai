"""Image classification endpoint."""

from __future__ import annotations

import asyncio
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
from backend.database.database import Database
from backend.database.repositories import insert_prediction
from backend.schemas.prediction import PredictionResponse
from backend.services.advice_service import AdviceService
from backend.services.image_service import (
    delete_prediction_images,
    save_prediction_images,
    validate_image_bytes,
)
from backend.services.model_service import ModelService
from ml.utils import utc_now_iso

router = APIRouter(tags=["predictions"])


def _analyze(
    settings: Settings,
    database: Database,
    model_service: ModelService,
    advice_service: AdviceService,
    data: bytes,
    filename: str | None,
) -> PredictionResponse:
    validated = validate_image_bytes(data, filename, settings.max_upload_bytes)
    result = model_service.predict(validated.image, validated.display)

    public_id = uuid4().hex
    created_at = utc_now_iso()
    try:
        files = save_prediction_images(
            settings.upload_dir, public_id, validated, result.gradcam
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
                "original_file": files["original_file"],
                "heatmap_file": files["heatmap_file"],
                "overlay_file": files["overlay_file"],
                "created_at": created_at,
            },
        )
    except Exception:
        delete_prediction_images(settings.upload_dir, public_id)
        raise

    media = f"{MEDIA_PREFIX}/{public_id}"
    return PredictionResponse(
        prediction_id=public_id,
        filename=validated.filename,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        top3=result.top3,
        advice=advice_service.for_class(result.predicted_class),
        gradcam={
            "original_url": f"{media}/{files['original_file']}",
            "heatmap_url": f"{media}/{files['heatmap_file']}",
            "overlay_url": f"{media}/{files['overlay_file']}",
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
                    _analyze,
                    settings,
                    database,
                    model_service,
                    advice_service,
                    data,
                    filename,
                )
    except TimeoutError as exc:
        raise AppError(
            504,
            "INFERENCE_TIMEOUT",
            "The analysis took too long. Please try again.",
        ) from exc
