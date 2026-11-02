from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.main import create_app
from backend.services import artifact_service
from ml.artifacts import load_checkpoint
from ml.classes import CLASS_NAMES
from ml.config import INPUT_SIZE
from ml.remote import ArtifactError, download_model_bundle
from ml.utils import read_json, write_json


def _write_checkpoint(path: Path, *, stage: str, arch: str = "custom_cnn") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "arch": arch,
            "class_names": list(CLASS_NAMES),
            "input_size": INPUT_SIZE,
            "stage": stage,
            "model_version": "test-version",
            "state_dict": {"weight": torch.zeros(2)},
        },
        path,
    )
    return path


class FakeHub:
    """Stands in for ``huggingface_hub.hf_hub_download``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, repo_id: str, filename: str, token: str | None) -> Path:
        self.calls.append((repo_id, filename, token))
        candidate = self.root / filename
        if not candidate.exists():
            raise FileNotFoundError(filename)
        return candidate


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    fake = FakeHub(root)
    monkeypatch.setattr("ml.remote._hub_download", fake)
    return fake


def test_download_installs_checkpoint_manifest_and_metrics(tmp_path, hub):
    _write_checkpoint(hub.root / "mobilenet_v2_final.pt", stage="final")
    write_json(hub.root / "metrics.json", {"accuracy": 0.9})

    models_dir = tmp_path / "models"
    evaluation_dir = tmp_path / "evaluation"

    installed = download_model_bundle(
        "org/ecovision",
        "mobilenet_v2_final.pt",
        models_dir,
        evaluation_dir,
        token="hf_test",
    )

    assert installed == models_dir / "mobilenet_v2_final.pt"
    assert load_checkpoint(installed)["stage"] == "final"
    assert hub.calls[0] == ("org/ecovision", "mobilenet_v2_final.pt", "hf_test")

    manifest = read_json(models_dir / "model_manifest.json")
    assert manifest["active"] == "mobilenet_v2_final.pt"
    assert manifest["stage"] == "final"
    assert manifest["arch"] == "custom_cnn"
    assert manifest["class_names"] == list(CLASS_NAMES)
    assert manifest["metrics_path"] == "evaluation/final/metrics.json"
    metrics = read_json(evaluation_dir / "final" / "metrics.json")
    assert metrics == {"accuracy": 0.9}


def test_checkpoint_stage_beats_misleading_manifest(tmp_path, hub):
    _write_checkpoint(hub.root / "mobilenet_v2_final.pt", stage="smoke")
    (hub.root / "model_manifest.json").write_text(
        '{"stage": "final", "arch": "mobilenet_v2"}', encoding="utf-8"
    )

    models_dir = tmp_path / "models"
    evaluation_dir = tmp_path / "evaluation"
    download_model_bundle(
        "org/ecovision",
        "mobilenet_v2_final.pt",
        models_dir,
        evaluation_dir,
    )

    manifest = read_json(models_dir / "model_manifest.json")
    assert manifest["stage"] == "smoke"
    assert manifest["arch"] == "custom_cnn"


def test_existing_valid_checkpoint_is_reused(tmp_path, hub):
    models_dir = tmp_path / "models"
    existing = _write_checkpoint(
        models_dir / "mobilenet_v2_final.pt", stage="final"
    )

    installed = download_model_bundle(
        "org/ecovision",
        "mobilenet_v2_final.pt",
        models_dir,
        tmp_path / "evaluation",
    )

    assert installed == existing
    assert hub.calls == []


def test_unsafe_artifact_paths_are_rejected(tmp_path, hub):
    with pytest.raises(ArtifactError):
        download_model_bundle(
            "org/ecovision",
            "../escape.pt",
            tmp_path / "models",
            tmp_path / "evaluation",
        )
    assert hub.calls == []


def test_missing_checkpoint_raises(tmp_path, hub):
    with pytest.raises(ArtifactError):
        download_model_bundle(
            "org/ecovision",
            "missing.pt",
            tmp_path / "models",
            tmp_path / "evaluation",
        )


def test_explicit_model_path_skips_download(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "local.pt", stage="final")
    settings = Settings(
        model_path=checkpoint,
        model_dir=tmp_path / "models",
        evaluation_dir=tmp_path / "evaluation",
        data_dir=tmp_path / "data",
        hf_repo="org/ecovision",
    )

    def explode(*args, **kwargs):
        raise AssertionError("remote download must be skipped")

    monkeypatch.setattr(artifact_service, "download_model_bundle", explode)

    assert artifact_service.fetch_model_artifacts(settings) is None


def test_empty_repo_or_disabled_fetch_skips_download(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("remote download must be skipped")

    monkeypatch.setattr(artifact_service, "download_model_bundle", explode)

    empty = Settings(
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        evaluation_dir=tmp_path / "evaluation",
    )
    disabled = Settings(
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        evaluation_dir=tmp_path / "evaluation",
        hf_repo="org/ecovision",
        fetch_model=False,
    )

    assert artifact_service.fetch_model_artifacts(empty) is None
    assert artifact_service.fetch_model_artifacts(disabled) is None


def test_download_failure_degrades_instead_of_crashing(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ArtifactError("hub is unreachable")

    monkeypatch.setattr(artifact_service, "download_model_bundle", fail)
    settings = Settings(
        model_dir=tmp_path / "models",
        evaluation_dir=tmp_path / "evaluation",
        data_dir=tmp_path / "data",
        hf_repo="org/ecovision",
        serve_frontend=False,
        log_level="ERROR",
        app_env="test",
    )

    app = create_app(settings)
    with TestClient(app) as client:
        payload = client.get("/api/health").json()
    assert payload["model_loaded"] is False
    assert payload["status"] == "degraded"


def test_startup_installs_remote_checkpoint(synthetic_checkpoint, tmp_path, monkeypatch):
    hub_root = tmp_path / "hub"
    hub_root.mkdir()
    target = hub_root / "mobilenet_v2_final.pt"
    shutil.copy(synthetic_checkpoint, target)

    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload["stage"] = "final"
    torch.save(payload, target)

    monkeypatch.setattr("ml.remote._hub_download", FakeHub(hub_root))

    settings = Settings(
        model_dir=tmp_path / "models",
        evaluation_dir=tmp_path / "evaluation",
        data_dir=tmp_path / "data",
        hf_repo="org/ecovision",
        serve_frontend=False,
        device="cpu",
        log_level="ERROR",
        app_env="test",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/api/health").json()

    assert health["model_loaded"] is True
    assert health["model_stage"] == "final"
    assert (tmp_path / "models" / "mobilenet_v2_final.pt").exists()
    manifest = read_json(tmp_path / "models" / "model_manifest.json")
    assert manifest["active"] == "mobilenet_v2_final.pt"
