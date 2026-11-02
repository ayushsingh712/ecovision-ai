from __future__ import annotations

import pytest

from ml.classes import CLASS_NAMES
from tests.helpers import image_bytes, upload


def _gradcam(client):
    loaded = client.app.state.model_service.loaded
    assert loaded is not None
    return loaded.gradcam


def _explain(client, prediction_id: str):
    return client.post(f"/api/predictions/{prediction_id}/explanation")


def test_classification_never_runs_gradcam(client, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Grad-CAM must not run on the classification path")

    monkeypatch.setattr(_gradcam(client), "generate", forbidden)

    response = upload(client, image_bytes())
    assert response.status_code == 200, response.text
    assert response.json()["top3"]


def test_explanation_generates_outputs_and_caches_them(
    client, sample_prediction, monkeypatch
):
    prediction_id = sample_prediction["prediction_id"]
    folder = client.app.state.settings.upload_dir / prediction_id
    assert not (folder / "heatmap.png").exists()

    first = _explain(client, prediction_id)
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["status"] == "ready"
    assert payload["gradcam"] == sample_prediction["gradcam"]
    assert (folder / "heatmap.png").is_file()
    assert (folder / "overlay.png").is_file()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Existing explanation files should be reused")

    monkeypatch.setattr(_gradcam(client), "generate", forbidden)
    cached = _explain(client, prediction_id)
    assert cached.status_code == 200, cached.text
    assert cached.json() == payload


def test_gradcam_failure_is_isolated_from_classification(client, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("synthetic Grad-CAM failure")

    monkeypatch.setattr(_gradcam(client), "generate", broken)

    response = upload(client, image_bytes())
    assert response.status_code == 200, response.text
    prediction = response.json()
    assert prediction["predicted_class"] in CLASS_NAMES
    assert prediction["top3"]
    assert prediction["advice"]["bin_type"]
    assert prediction["gradcam"]["heatmap_url"].startswith("/media/")

    explanation = _explain(client, prediction["prediction_id"])
    assert explanation.status_code == 200, explanation.text
    payload = explanation.json()
    assert payload["status"] == "failed"
    assert payload["gradcam"] is None
    assert "still valid" in payload["message"]

    folder = client.app.state.settings.upload_dir / prediction["prediction_id"]
    assert not (folder / "heatmap.png").exists()

    history = client.get("/api/history").json()
    item = next(
        row
        for row in history["items"]
        if row["prediction_id"] == prediction["prediction_id"]
    )
    assert item["predicted_class"] == prediction["predicted_class"]


def test_explanation_can_be_disabled(client, sample_prediction):
    client.app.state.settings.gradcam_enabled = False

    response = _explain(client, sample_prediction["prediction_id"])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "disabled"
    assert payload["gradcam"] is None

    folder = client.app.state.settings.upload_dir / sample_prediction["prediction_id"]
    assert not (folder / "heatmap.png").exists()


def test_explanation_for_unknown_prediction_returns_404(client):
    response = _explain(client, "0" * 32)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PREDICTION_NOT_FOUND"


def test_explanation_reports_missing_stored_image(client, sample_prediction):
    folder = client.app.state.settings.upload_dir / sample_prediction["prediction_id"]
    (folder / "original.jpg").unlink()

    response = _explain(client, sample_prediction["prediction_id"])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "failed"
    assert "no longer available" in payload["message"]


@pytest.mark.parametrize("prediction_id", ["", "../../etc/passwd"])
def test_explanation_rejects_unsafe_prediction_ids(client, prediction_id):
    response = _explain(client, prediction_id)
    assert response.status_code == 404
