"""SQL queries used by the API services."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from backend.database.database import Database


def insert_prediction(db: Database, record: dict[str, Any]) -> int:
    top3_json = json.dumps(record["top3"], separators=(",", ":"))
    with db.connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                public_id, original_filename, mime_type, size_bytes, sha256,
                width, height, predicted_class, confidence, top3_json,
                model_stage, model_version, original_file, heatmap_file,
                overlay_file, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["public_id"],
                record["original_filename"],
                record["mime_type"],
                record["size_bytes"],
                record["sha256"],
                record["width"],
                record["height"],
                record["predicted_class"],
                record["confidence"],
                top3_json,
                record["model_stage"],
                record["model_version"],
                record["original_file"],
                record["heatmap_file"],
                record["overlay_file"],
                record["created_at"],
            ),
        )
        return int(cursor.lastrowid)


def get_prediction_by_public_id(
    db: Database, public_id: str
) -> sqlite3.Row | None:
    with db.connection() as connection:
        return connection.execute(
            "SELECT * FROM predictions WHERE public_id = ?", (public_id,)
        ).fetchone()


def update_prediction_explanation(
    db: Database,
    public_id: str,
    heatmap_file: str,
    overlay_file: str,
) -> None:
    with db.connection() as connection:
        connection.execute(
            """
            UPDATE predictions
            SET heatmap_file = ?, overlay_file = ?
            WHERE public_id = ?
            """,
            (heatmap_file, overlay_file, public_id),
        )


def upsert_feedback(
    db: Database,
    prediction_id: int,
    is_correct: bool,
    corrected_class: str | None,
    comment: str | None,
    created_at: str,
) -> sqlite3.Row:
    with db.connection() as connection:
        connection.execute(
            """
            INSERT INTO feedback (
                prediction_id, is_correct, corrected_class, comment,
                verified, created_at
            ) VALUES (?, ?, ?, ?, 0, ?)
            ON CONFLICT(prediction_id) DO UPDATE SET
                is_correct = excluded.is_correct,
                corrected_class = excluded.corrected_class,
                comment = excluded.comment,
                verified = 0,
                created_at = excluded.created_at
            """,
            (prediction_id, int(is_correct), corrected_class, comment, created_at),
        )
        row = connection.execute(
            "SELECT * FROM feedback WHERE prediction_id = ?", (prediction_id,)
        ).fetchone()
    assert row is not None
    return row


def fetch_history(
    db: Database, limit: int, offset: int
) -> tuple[list[sqlite3.Row], int]:
    with db.connection() as connection:
        total = int(
            connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT
                p.*,
                f.is_correct AS feedback_is_correct,
                f.corrected_class AS feedback_corrected_class,
                f.comment AS feedback_comment,
                f.verified AS feedback_verified,
                f.created_at AS feedback_created_at
            FROM predictions p
            LEFT JOIN feedback f ON f.prediction_id = p.id
            ORDER BY p.created_at DESC, p.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return list(rows), total


def fetch_class_distribution(db: Database) -> list[sqlite3.Row]:
    with db.connection() as connection:
        return list(
            connection.execute(
                """
                SELECT predicted_class, COUNT(*) AS count
                FROM predictions
                GROUP BY predicted_class
                ORDER BY count DESC
                """
            ).fetchall()
        )


def fetch_summary(db: Database) -> dict[str, Any]:
    with db.connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total, AVG(confidence) AS average FROM predictions"
        ).fetchone()
        feedback = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(is_correct), 0) AS correct
            FROM feedback
            """
        ).fetchone()
        daily = connection.execute(
            """
            SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS count
            FROM predictions
            WHERE date(created_at) >= date('now', '-6 days')
            GROUP BY substr(created_at, 1, 10)
            ORDER BY date ASC
            """
        ).fetchall()
    return {
        "total": int(row["total"] or 0),
        "average_confidence": row["average"],
        "feedback_total": int(feedback["total"] or 0),
        "feedback_correct": int(feedback["correct"] or 0),
        "daily": [dict(item) for item in daily],
    }
