"""Model, dataset, and evaluation metadata endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.deps import AdviceServiceDep, ModelServiceDep
from backend.schemas.model_info import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    model_service: ModelServiceDep, advice_service: AdviceServiceDep
) -> ModelInfoResponse:
    payload = model_service.model_info()
    payload["advice_version"] = advice_service.version
    payload["advice_last_reviewed"] = advice_service.last_reviewed
    return ModelInfoResponse.model_validate(payload)
