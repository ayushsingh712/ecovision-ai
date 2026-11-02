"""Aggregate dashboard statistics endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.deps import DatabaseDep, ModelServiceDep
from backend.database.repositories import fetch_class_distribution, fetch_summary
from backend.schemas.statistics import StatisticsResponse
from backend.services.statistics_service import build_statistics

router = APIRouter(tags=["statistics"])


@router.get("/statistics", response_model=StatisticsResponse)
def statistics(
    database: DatabaseDep, model_service: ModelServiceDep
) -> StatisticsResponse:
    loaded = model_service.loaded
    payload = build_statistics(
        fetch_summary(database),
        fetch_class_distribution(database),
        loaded.stage if loaded else None,
    )
    return StatisticsResponse.model_validate(payload)
