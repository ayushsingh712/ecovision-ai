"""Feedback request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from ml.classes import CLASS_NAMES


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(min_length=8, max_length=64)
    is_correct: bool
    corrected_class: str | None = None
    comment: str | None = Field(default=None, max_length=1000)

    @field_validator("corrected_class")
    @classmethod
    def _validate_class(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in CLASS_NAMES:
            raise ValueError(f"Corrected class must be one of: {', '.join(CLASS_NAMES)}")
        return value

    @field_validator("comment")
    @classmethod
    def _clean_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def _consistent_correction(self):
        if self.is_correct and self.corrected_class:
            raise ValueError("corrected_class must be omitted when is_correct is true.")
        return self


class FeedbackResponse(BaseModel):
    prediction_id: str
    is_correct: bool
    corrected_class: str | None
    comment: str | None
    verified: bool
    created_at: str
