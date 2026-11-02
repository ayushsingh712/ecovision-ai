"""Download deployment model artifacts from the Hugging Face Hub.

Checkpoints are never committed. A deployment points ``ECOVISION_HF_REPO`` at a
model repository and this module installs the checkpoint, manifest, and (when
published) evaluation metrics into the expected local layout.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from ml.artifacts import load_checkpoint
from ml.utils import read_json, write_json

logger = logging.getLogger("ecovision.artifacts")


class ArtifactError(RuntimeError):
    """Raised when a deployment artifact cannot be installed."""


def _safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ArtifactError(f"Refusing unsafe artifact path: {name!r}")
    return path


def _copy_atomic(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(handle)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _hub_download(repo_id: str, filename: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    )


def _download_optional(
    repo_id: str, filename: str | None, token: str | None
) -> Path | None:
    if not filename:
        return None
    try:
        return _hub_download(repo_id, filename, token)
    except Exception as exc:
        logger.info("Optional artifact %s is not available: %s", filename, exc)
        return None


def download_model_bundle(
    repo_id: str,
    filename: str,
    models_dir: Path,
    evaluation_dir: Path,
    *,
    manifest_filename: str | None = "model_manifest.json",
    metrics_filename: str | None = "metrics.json",
    token: str | None = None,
) -> Path:
    """Install a checkpoint (plus optional manifest/metrics) from the Hub.

    Returns the local checkpoint path. Raises :class:`ArtifactError` when the
    checkpoint itself cannot be downloaded or validated.
    """
    if not repo_id.strip():
        raise ArtifactError("A Hugging Face repository id is required.")

    relative = _safe_relative(filename)
    target = models_dir / relative

    if target.exists():
        try:
            payload = load_checkpoint(target)
            logger.info("Using existing model checkpoint %s", target)
            return target
        except Exception as exc:
            logger.warning("Existing checkpoint %s is invalid (%s); replacing", target, exc)
            target.unlink(missing_ok=True)

    try:
        cached = _hub_download(repo_id, relative.as_posix(), token)
    except Exception as exc:
        raise ArtifactError(
            f"Could not download {relative.as_posix()} from {repo_id}: {exc}"
        ) from exc

    payload: dict[str, Any] = load_checkpoint(cached)
    _copy_atomic(cached, target)

    manifest_source = _download_optional(repo_id, manifest_filename, token)
    manifest: dict[str, Any] = {}
    if manifest_source is not None:
        manifest = read_json(manifest_source, {}) or {}
        if not isinstance(manifest, dict):
            manifest = {}

    stage = str(payload.get("stage") or manifest.get("stage") or "final")
    arch = str(payload.get("arch") or manifest.get("arch") or "")
    metrics_path: str | None = None

    metrics_source = _download_optional(repo_id, metrics_filename, token)
    if metrics_source is not None:
        metrics_target = evaluation_dir / stage / "metrics.json"
        _copy_atomic(metrics_source, metrics_target)
        metrics_path = str(metrics_target.relative_to(evaluation_dir.parent))

    manifest.update(
        {
            "active": relative.as_posix(),
            "stage": stage,
            "arch": arch,
            "model_version": manifest.get("model_version")
            or payload.get("model_version")
            or "unknown",
            "class_names": list(payload["class_names"]),
            "metrics_path": manifest.get("metrics_path") or metrics_path,
        }
    )
    write_json(models_dir / "model_manifest.json", manifest)

    logger.info(
        "Installed model artifact %s from %s (stage=%s, metrics=%s)",
        relative.as_posix(),
        repo_id,
        stage,
        "yes" if metrics_source is not None else "no",
    )
    return target
