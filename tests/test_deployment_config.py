from __future__ import annotations

from backend.core.config import PROJECT_ROOT, Settings


def test_storage_paths_derive_from_data_dir(tmp_path):
    data_dir = tmp_path / "runtime"
    settings = Settings(
        data_dir=data_dir,
        model_dir=tmp_path / "models",
        evaluation_dir=tmp_path / "evaluation",
    )

    assert settings.db_path == data_dir / "ecovision.db"
    assert settings.upload_dir == data_dir / "uploads"


def test_explicit_storage_paths_win(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "runtime",
        db_path=tmp_path / "custom" / "app.db",
        upload_dir=tmp_path / "custom" / "media",
    )

    assert settings.db_path == tmp_path / "custom" / "app.db"
    assert settings.upload_dir == tmp_path / "custom" / "media"


def test_relative_paths_resolve_against_project_root(tmp_path):
    settings = Settings(data_dir="var/runtime", evaluation_dir="var/evaluation")

    assert settings.data_dir == (PROJECT_ROOT / "var/runtime").resolve()
    assert settings.evaluation_dir == (PROJECT_ROOT / "var/evaluation").resolve()
    assert settings.db_path == settings.data_dir / "ecovision.db"


def test_remote_model_is_enabled_by_default():
    settings = Settings(hf_repo="org/ecovision")

    assert settings.fetch_model is True
    assert settings.hf_filename == "mobilenet_v2_final.pt"
