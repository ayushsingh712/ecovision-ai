from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.main import create_app
from ml.architectures.factory import build_model
from ml.artifacts import build_checkpoint, save_checkpoint
from ml.classes import CLASS_NAMES
from tests.helpers import image_bytes, upload


@pytest.fixture(scope="session")
def synthetic_checkpoint(tmp_path_factory):
    """A real architecture with seeded random weights, clearly marked as test-only."""
    directory = tmp_path_factory.mktemp("model")
    model = build_model("custom_cnn", len(CLASS_NAMES), pretrained=False)
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    payload = build_checkpoint(
        model=model,
        arch="custom_cnn",
        stage="test",
        history=[],
        split_hash="test-split-hash",
        extra={"synthetic": True},
    )
    path = directory / "custom_cnn_test.pt"
    save_checkpoint(path, payload)
    return path


@pytest.fixture()
def client(synthetic_checkpoint, tmp_path):
    settings = Settings(
        model_path=synthetic_checkpoint,
        model_dir=synthetic_checkpoint.parent,
        db_path=tmp_path / "test.db",
        upload_dir=tmp_path / "uploads",
        serve_frontend=False,
        device="cpu",
        log_level="ERROR",
        app_env="test",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def unavailable_client(tmp_path):
    settings = Settings(
        model_path=tmp_path / "missing.pt",
        model_dir=tmp_path,
        db_path=tmp_path / "test.db",
        upload_dir=tmp_path / "uploads",
        serve_frontend=False,
        device="cpu",
        log_level="ERROR",
        app_env="test",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def jpeg_image() -> bytes:
    return image_bytes("JPEG")


@pytest.fixture()
def sample_prediction(client, jpeg_image):
    response = upload(client, jpeg_image)
    assert response.status_code == 200, response.text
    return response.json()
