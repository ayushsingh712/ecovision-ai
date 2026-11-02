"""Formatting helpers for dashboard statistics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ml.classes import CLASS_NAMES


def build_statistics(
    summary: dict[str, Any],
    distribution_rows: list[Any],
    model_stage: str | None,
) -> dict[str, Any]:
    total = int(summary["total"])
    counts = {row["predicted_class"]: int(row["count"]) for row in distribution_rows}
    distribution = [
        {
            "label": label,
            "count": counts.get(label, 0),
            "percentage": round(counts.get(label, 0) / total * 100, 2) if total else 0.0,
        }
        for label in CLASS_NAMES
    ]

    feedback_total = int(summary["feedback_total"])
    feedback_correct = int(summary["feedback_correct"])
    daily_lookup = {
        row["date"]: int(row["count"]) for row in summary.get("daily", [])
    }
    today = datetime.now(timezone.utc).date()
    daily = []
    for offset in range(6, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        daily.append({"date": day, "count": daily_lookup.get(day, 0)})

    counts_only = [item for item in distribution if item["count"] > 0]
    most_common = max(counts_only, key=lambda item: item["count"])["label"] if counts_only else None
    average = summary["average_confidence"]
    return {
        "total_predictions": total,
        "average_confidence": round(float(average), 4) if average is not None else None,
        "most_common_class": most_common,
        "class_distribution": distribution,
        "feedback": {
            "total": feedback_total,
            "correct": feedback_correct,
            "incorrect": max(feedback_total - feedback_correct, 0),
            "user_reported_accuracy": (
                round(feedback_correct / feedback_total, 4) if feedback_total else None
            ),
        },
        "daily": daily,
        "model_stage": model_stage,
    }
