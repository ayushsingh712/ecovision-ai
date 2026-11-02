from __future__ import annotations

import pytest

from backend.core.errors import AppError
from backend.services.image_service import (
    sanitize_filename,
    validate_image_bytes,
)
from tests.helpers import image_bytes


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_supported_formats_are_accepted(fmt, mime):
    validated = validate_image_bytes(image_bytes(fmt), f"photo.{fmt.lower()}", 8 * 1024 * 1024)
    assert validated.mime_type == mime
    assert validated.image.mode == "RGB"
    assert validated.display.size == (512, 512)
    assert validated.width == 160
    assert len(validated.sha256) == 64


def test_empty_file_rejected():
    with pytest.raises(AppError) as excinfo:
        validate_image_bytes(b"", "empty.jpg", 8 * 1024 * 1024)
    assert excinfo.value.status_code == 422
    assert excinfo.value.code == "EMPTY_FILE"


def test_text_masquerading_as_image_rejected():
    with pytest.raises(AppError) as excinfo:
        validate_image_bytes(b"this is not an image", "notes.jpg", 8 * 1024 * 1024)
    assert excinfo.value.status_code == 415
    assert excinfo.value.code == "UNSUPPORTED_IMAGE_TYPE"


def test_truncated_jpeg_rejected():
    truncated = image_bytes("JPEG")[:60]
    with pytest.raises(AppError):
        validate_image_bytes(truncated, "broken.jpg", 8 * 1024 * 1024)


def test_tiny_image_rejected():
    with pytest.raises(AppError) as excinfo:
        validate_image_bytes(image_bytes("PNG", (1, 1)), "tiny.png", 8 * 1024 * 1024)
    assert excinfo.value.code == "IMAGE_TOO_SMALL"


def test_size_limit_enforced_before_decoding():
    data = b"\xff\xd8\xff" + b"0" * 200
    with pytest.raises(AppError) as excinfo:
        validate_image_bytes(data, "big.jpg", 100)
    assert excinfo.value.status_code == 413
    assert excinfo.value.code == "FILE_TOO_LARGE"


def test_filename_sanitized():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("a\x00b.jpg") == "ab.jpg"
    assert sanitize_filename(None) == "upload"
    assert len(sanitize_filename("x" * 500)) == 120
