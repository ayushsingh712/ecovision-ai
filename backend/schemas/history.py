"""Prediction history contracts."""

from __future__ import annotations

from pydantic import BaseModel

from backend.schemas.prediction import ClassScore


class FeedbackInfo(BaseModel):
    is_correct: bool
    corrected_class: str | None
    comment: str | None
    verified: bool
    created_at: str


class HistoryItem(BaseModel):
    prediction_id: str
    filename: str
    predicted_class: str
    confidence: float
    top3: list[ClassScore]
    image_url: str
    created_at: str
    model_stage: str
    feedback: FeedbackInfo | None


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int
