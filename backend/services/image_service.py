"""Upload validation, normalization, and media persistence."""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from backend.core.errors import AppError

ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MIN_SIDE = 64
MAX_PIXELS = 40_000_000
DISPLAY_SIZE = 512
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class ValidatedImage:
    image: Image.Image
    display: Image.Image
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    width: int
    height: int


def sanitize_filename(value: str | None) -> str:
    name = Path(value or "upload").name
    name = CONTROL_CHARS.sub("", name).strip()
    return (name or "upload")[:120]


def _detect_format(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def _square_display(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - 224) // 2)
    top = max(0, (resized.height - 224) // 2)
    cropped = resized.crop((left, top, left + 224, top + 224))
    return cropped.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.LANCZOS)


def validate_image_bytes(
    data: bytes, filename: str | None, max_bytes: int
) -> ValidatedImage:
    if not data:
        raise AppError(422, "EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise AppError(
            413,
            "FILE_TOO_LARGE",
            f"Images must be {max_bytes // (1024 * 1024)} MB or smaller.",
        )

    detected = _detect_format(data)
    if detected is None:
        raise AppError(
            415,
            "UNSUPPORTED_IMAGE_TYPE",
            "Only JPEG, PNG, and WebP images are supported.",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            if image.format != detected:
                raise AppError(
                    415,
                    "UNSUPPORTED_IMAGE_TYPE",
                    "The file contents do not match a supported image format.",
                )
            if getattr(image, "n_frames", 1) > 1:
                raise AppError(
                    422,
                    "ANIMATED_IMAGE",
                    "Animated images are not supported.",
                )
            width, height = image.size
            if width < MIN_SIDE or height < MIN_SIDE:
                raise AppError(
                    422,
                    "IMAGE_TOO_SMALL",
                    f"Images must be at least {MIN_SIDE}x{MIN_SIDE} pixels.",
                )
            if width * height > MAX_PIXELS:
                raise AppError(
                    422,
                    "IMAGE_TOO_LARGE_PIXELS",
                    "The image resolution is too large to process safely.",
                )
            normalized = ImageOps.exif_transpose(image).convert("RGB")
    except AppError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise AppError(422, "INVALID_IMAGE", "The image could not be read.") from exc

    return ValidatedImage(
        image=normalized,
        display=_square_display(normalized),
        filename=sanitize_filename(filename),
        mime_type=ALLOWED_FORMATS[detected],
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
    )


def save_prediction_images(
    upload_dir: Path, public_id: str, validated: ValidatedImage, gradcam
) -> dict[str, str]:
    target = upload_dir / public_id
    target.mkdir(parents=True, exist_ok=True)
    names = {
        "original_file": "original.jpg",
        "heatmap_file": "heatmap.png",
        "overlay_file": "overlay.png",
    }
    try:
        validated.display.save(
            target / names["original_file"], format="JPEG", quality=88, optimize=True
        )
        gradcam.heatmap.save(target / names["heatmap_file"], format="PNG", optimize=True)
        gradcam.overlay.save(target / names["overlay_file"], format="PNG", optimize=True)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return names


def delete_prediction_images(upload_dir: Path, public_id: str) -> None:
    shutil.rmtree(upload_dir / public_id, ignore_errors=True)
