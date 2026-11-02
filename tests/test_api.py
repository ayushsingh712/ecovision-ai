from __future__ import annotations

from ml.classes import CLASS_NAMES


def test_health_reports_loaded_model(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_stage"] == "test"
    assert payload["database"] == "ok"
    assert payload["device"] == "cpu"
    assert response.headers["X-Request-ID"]


def test_health_degraded_without_checkpoint(unavailable_client):
    response = unavailable_client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["model_loaded"] is False


def test_model_info_contract(client):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert [item["name"] for item in payload["classes"]] == list(CLASS_NAMES)
    assert len(payload["architectures"]) == 2
    names = {item["name"] for item in payload["architectures"]}
    assert names == {"custom_cnn", "mobilenet_v2"}
    assert all(item["parameters"] > 0 for item in payload["architectures"])
    assert payload["dataset"]["total_images"] == 4752
    assert payload["dataset"]["license"] == "CC BY 4.0"
    assert payload["architecture"]["parameters"] > 0
    assert payload["model_stage"] == "test"
    assert payload["is_final"] is False
    assert payload["limitations"]
    assert payload["advice_version"]
    assert payload["advice_last_reviewed"]


def test_unknown_api_endpoint_returns_json_error(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "NOT_FOUND"
    assert error["request_id"]


def test_history_defaults(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_statistics_empty_state(client):
    response = client.get("/api/statistics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_predictions"] == 0
    assert len(payload["class_distribution"]) == 7
    assert all(item["count"] == 0 for item in payload["class_distribution"])
    assert len(payload["daily"]) == 7
    assert payload["model_stage"] == "test"
