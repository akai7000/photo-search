"""Generate bounded, correctly oriented previews on demand."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


def create_photo_preview(photo_path: str | Path, size: tuple[int, int] = (1600, 1600)) -> str:
    with Image.open(photo_path) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail(size)
        image = image.convert("RGB")
        output = BytesIO()
        image.save(output, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
