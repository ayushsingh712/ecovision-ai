from __future__ import annotations

from tests.helpers import image_bytes, upload


def test_unsupported_content_rejected(client):
    response = upload(client, b"plain text pretending to be a jpeg", "notes.jpg")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_IMAGE_TYPE"


def test_empty_upload_rejected(client):
    response = upload(client, b"")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "EMPTY_FILE"


def test_tiny_image_rejected(client):
    response = upload(client, image_bytes("PNG", (1, 1)), "tiny.png", "image/png")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "IMAGE_TOO_SMALL"


def test_truncated_image_rejected(client):
    response = upload(client, image_bytes("JPEG")[:70], "broken.jpg")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_oversized_upload_rejected(client):
    oversized = b"\xff\xd8\xff" + b"0" * (8 * 1024 * 1024)
    response = upload(client, oversized, "huge.jpg")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_missing_image_field_rejected(client):
    response = client.post("/api/predict")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_feedback_for_unknown_prediction(client):
    response = client.post(
        "/api/feedback",
        json={"prediction_id": "00000000-0000-0000-0000-000000000000", "is_correct": True},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PREDICTION_NOT_FOUND"


def test_feedback_rejects_unknown_class(client, sample_prediction):
    response = client.post(
        "/api/feedback",
        json={
            "prediction_id": sample_prediction["prediction_id"],
            "is_correct": False,
            "corrected_class": "Compost",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_feedback_rejects_correction_when_marked_correct(client, sample_prediction):
    response = client.post(
        "/api/feedback",
        json={
            "prediction_id": sample_prediction["prediction_id"],
            "is_correct": True,
            "corrected_class": "Glass",
        },
    )
    assert response.status_code == 422


def test_history_validates_pagination(client):
    assert client.get("/api/history?limit=0").status_code == 422
    assert client.get("/api/history?limit=101").status_code == 422
    assert client.get("/api/history?offset=-1").status_code == 422
