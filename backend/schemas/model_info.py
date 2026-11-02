"""Model information contracts for the About page."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ArchitectureInfo(BaseModel):
    name: str
    display_name: str
    parameters: int
    input_size: int
    target_layer: str
    description: str


class ClassInfo(BaseModel):
    index: int
    name: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_name: str | None
    model_stage: str | None
    model_version: str | None
    is_final: bool
    device: str | None
    checkpoint: str | None
    trained_at: str | None
    architecture: ArchitectureInfo | None
    architectures: list[ArchitectureInfo]
    classes: list[ClassInfo]
    dataset: dict[str, Any]
    metrics: dict[str, Any] | None
    comparison: list[dict[str, Any]] | None
    limitations: list[str]
    advice_version: str
    advice_last_reviewed: str
