"""Service health endpoint."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from backend.api.deps import DatabaseDep, ModelServiceDep, SettingsDep
from backend.core.config import APP_VERSION
from backend.schemas.common import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    settings: SettingsDep,
    database: DatabaseDep,
    model_service: ModelServiceDep,
) -> HealthResponse:
    started_at = getattr(request.app.state, "started_at", time.monotonic())
    loaded = model_service.loaded
    return HealthResponse(
        status="ok" if model_service.available else "degraded",
        app=settings.app_name,
        version=APP_VERSION,
        environment=settings.app_env,
        model_loaded=model_service.available,
        model_stage=loaded.stage if loaded else None,
        model_version=loaded.version if loaded else None,
        device=model_service.device if loaded else None,
        database="ok" if database.ping() else "error",
        uptime_seconds=round(time.monotonic() - started_at, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
