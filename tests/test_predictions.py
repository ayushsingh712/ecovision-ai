from __future__ import annotations

from uuid import UUID

from ml.classes import CLASS_NAMES
from tests.helpers import image_bytes, upload


def test_prediction_response_contract(sample_prediction):
    data = sample_prediction
    UUID(data["prediction_id"])
    assert data["predicted_class"] in CLASS_NAMES
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_stage"] == "test"
    assert data["model_version"]
    assert data["filename"] == "waste.jpg"

    top3 = data["top3"]
    assert len(top3) == 3
    assert top3[0]["label"] == data["predicted_class"]
    assert top3[0]["confidence"] == data["confidence"]
    confidences = [item["confidence"] for item in top3]
    assert confidences == sorted(confidences, reverse=True)
    assert sum(confidences) <= 1.000001
    assert data["low_confidence"] == (data["confidence"] < 0.6)

    advice = data["advice"]
    assert advice["bin_type"]
    assert advice["disposal_method"]
    assert advice["instructions"]
    assert advice["environmental_impact"]
    assert advice["scope"] == "general"
    assert advice["disclaimer"]
    assert advice["sources"]

    for key in ("original_url", "heatmap_url", "overlay_url"):
        assert data["gradcam"][key].startswith(f"/media/{data['prediction_id']}/")


def test_prediction_media_files_are_saved(client, sample_prediction):
    upload_dir = client.app.state.settings.upload_dir
    folder = upload_dir / sample_prediction["prediction_id"]
    assert (folder / "original.jpg").is_file()
    assert (folder / "heatmap.png").is_file()
    assert (folder / "overlay.png").is_file()

    response = client.get(sample_prediction["gradcam"]["original_url"])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")


def test_feedback_upserts_and_history_reflects_it(client, sample_prediction):
    prediction_id = sample_prediction["prediction_id"]

    response = client.post(
        "/api/feedback",
        json={"prediction_id": prediction_id, "is_correct": True},
    )
    assert response.status_code == 200
    assert response.json()["verified"] is False

    corrected = "Glass" if sample_prediction["predicted_class"] != "Glass" else "Metal"
    response = client.post(
        "/api/feedback",
        json={
            "prediction_id": prediction_id,
            "is_correct": False,
            "corrected_class": corrected,
            "comment": "  It is a jar.  ",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_correct"] is False
    assert payload["corrected_class"] == corrected
    assert payload["comment"] == "It is a jar."
    assert payload["verified"] is False

    history = client.get("/api/history").json()
    item = next(row for row in history["items"] if row["prediction_id"] == prediction_id)
    assert item["feedback"]["is_correct"] is False
    assert item["feedback"]["corrected_class"] == corrected

    stats = client.get("/api/statistics").json()
    assert stats["feedback"]["total"] == 1
    assert stats["feedback"]["correct"] == 0
    assert stats["feedback"]["incorrect"] == 1
    assert stats["feedback"]["user_reported_accuracy"] == 0.0


def test_history_pagination(client):
    for _ in range(3):
        response = upload(client, image_bytes("PNG"))
        assert response.status_code == 200, response.text

    first = client.get("/api/history?limit=1&offset=0").json()
    assert first["total"] == 3
    assert len(first["items"]) == 1

    second = client.get("/api/history?limit=1&offset=1").json()
    assert len(second["items"]) == 1
    assert second["items"][0]["prediction_id"] != first["items"][0]["prediction_id"]

    all_items = client.get("/api/history?limit=100").json()["items"]
    assert len(all_items) == 3
    for item in all_items:
        assert item["image_url"].startswith("/media/")


def test_statistics_aggregate(client):
    for fmt in ("JPEG", "PNG", "WEBP"):
        assert upload(client, image_bytes(fmt)).status_code == 200

    stats = client.get("/api/statistics").json()
    assert stats["total_predictions"] == 3
    assert stats["average_confidence"] is not None
    assert 0.0 <= stats["average_confidence"] <= 1.0
    assert len(stats["class_distribution"]) == 7
    assert sum(item["count"] for item in stats["class_distribution"]) == 3
    assert stats["most_common_class"] in CLASS_NAMES
    assert len(stats["daily"]) == 7
    assert stats["daily"][-1]["count"] == 3
    assert stats["model_stage"] == "test"


def test_prediction_without_model_returns_503(unavailable_client, jpeg_image):
    response = upload(unavailable_client, jpeg_image)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"
