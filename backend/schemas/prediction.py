"""Prediction response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class SourceLink(BaseModel):
    title: str
    url: str


class Advice(BaseModel):
    bin_type: str
    bin_color: str
    disposal_method: str
    instructions: list[str]
    environmental_impact: str
    examples: list[str]
    cautions: list[str]
    scope: str = "general"
    region_note: str
    disclaimer: str
    sources: list[SourceLink]


class GradCAMUrls(BaseModel):
    original_url: str
    heatmap_url: str
    overlay_url: str


class PredictionResponse(BaseModel):
    prediction_id: str
    filename: str
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top3: list[ClassScore]
    advice: Advice
    gradcam: GradCAMUrls
    model_stage: str
    model_version: str
    created_at: str
    low_confidence: bool
