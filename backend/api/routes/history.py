"""Paginated prediction history endpoint."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query

from backend.api.deps import DatabaseDep
from backend.core.constants import MEDIA_PREFIX
from backend.database.repositories import fetch_history
from backend.schemas.history import FeedbackInfo, HistoryItem, HistoryResponse

router = APIRouter(tags=["history"])


def _feedback(row: Any) -> FeedbackInfo | None:
    if row["feedback_is_correct"] is None:
        return None
    return FeedbackInfo(
        is_correct=bool(row["feedback_is_correct"]),
        corrected_class=row["feedback_corrected_class"],
        comment=row["feedback_comment"],
        verified=bool(row["feedback_verified"]),
        created_at=row["feedback_created_at"],
    )


@router.get("/history", response_model=HistoryResponse)
def history(
    database: DatabaseDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> HistoryResponse:
    rows, total = fetch_history(database, limit, offset)
    items = [
        HistoryItem(
            prediction_id=row["public_id"],
            filename=row["original_filename"],
            predicted_class=row["predicted_class"],
            confidence=round(float(row["confidence"]), 6),
            top3=json.loads(row["top3_json"]),
            image_url=f"{MEDIA_PREFIX}/{row['public_id']}/{row['original_file']}",
            created_at=row["created_at"],
            model_stage=row["model_stage"],
            feedback=_feedback(row),
        )
        for row in rows
    ]
    return HistoryResponse(items=items, total=total, limit=limit, offset=offset)
