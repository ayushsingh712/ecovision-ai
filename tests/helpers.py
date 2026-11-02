from __future__ import annotations

import io

from PIL import Image


def image_bytes(
    fmt: str = "JPEG", size: tuple[int, int] = (160, 160), color=(40, 120, 80)
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(client, data: bytes, filename: str = "waste.jpg", mime: str = "image/jpeg"):
    return client.post("/api/predict", files={"image": (filename, data, mime)})
