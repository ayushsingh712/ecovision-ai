"""User feedback endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.deps import DatabaseDep
from backend.core.errors import AppError
from backend.database.repositories import get_prediction_by_public_id, upsert_feedback
from backend.schemas.feedback import FeedbackRequest, FeedbackResponse
from ml.utils import utc_now_iso

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    payload: FeedbackRequest, database: DatabaseDep
) -> FeedbackResponse:
    prediction = get_prediction_by_public_id(database, payload.prediction_id)
    if prediction is None:
        raise AppError(
            404,
            "PREDICTION_NOT_FOUND",
            "No prediction exists with that id.",
        )

    row = upsert_feedback(
        database,
        prediction_id=int(prediction["id"]),
        is_correct=payload.is_correct,
        corrected_class=payload.corrected_class,
        comment=payload.comment,
        created_at=utc_now_iso(),
    )
    return FeedbackResponse(
        prediction_id=payload.prediction_id,
        is_correct=bool(row["is_correct"]),
        corrected_class=row["corrected_class"],
        comment=row["comment"],
        verified=bool(row["verified"]),
        created_at=row["created_at"],
    )
