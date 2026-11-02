"""Dashboard statistics contracts."""

from __future__ import annotations

from pydantic import BaseModel


class ClassCount(BaseModel):
    label: str
    count: int
    percentage: float


class FeedbackStats(BaseModel):
    total: int
    correct: int
    incorrect: int
    user_reported_accuracy: float | None


class DailyCount(BaseModel):
    date: str
    count: int


class StatisticsResponse(BaseModel):
    total_predictions: int
    average_confidence: float | None
    most_common_class: str | None
    class_distribution: list[ClassCount]
    feedback: FeedbackStats
    daily: list[DailyCount]
    model_stage: str | None
