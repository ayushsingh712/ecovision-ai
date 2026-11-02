"""Shared API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: Any | None = None
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    environment: str
    model_loaded: bool
    model_stage: str | None
    model_version: str | None
    device: str | None
    database: Literal["ok", "error"]
    uptime_seconds: float
    timestamp: str
