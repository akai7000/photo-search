import math
import os
import sqlite3
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from core.database import get_connection, DB_PATH


FACE_THUMBNAIL_DIR = DB_PATH.parent / "face_thumbnails"


def delete_face_thumbnails(face_ids: list[int]) -> list[str]:
    """Remove cached face images and return any filesystem errors."""
    errors = []
    for face_id in face_ids:
        for filename in (f"face_{face_id}_v2.jpg", f"face_{face_id}.jpg"):
            path = FACE_THUMBNAIL_DIR / filename
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                print(f"Could not remove face thumbnail {path}: {exc}")
    return errors


def ensure_face_thumbnails(
    face_ids: list[int],
    *,
    connection: sqlite3.Connection | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[int, Path], dict[int, str]]:
    """Create missing face examples from the coordinates stored during detection."""
    if not face_ids:
        return {}, {}

    output_dir = output_dir if output_dir is not None else FACE_THUMBNAIL_DIR
    own_connection = connection is None
    if connection is None:
        connection = get_connection()
    rows = []
    try:
        for offset in range(0, len(face_ids), 500):
            batch = face_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(connection.execute(f"""
                SELECT f.id AS face_id, p.path AS photo_path,
                       f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h
                FROM faces f JOIN photos p ON p.id = f.photo_id
                WHERE f.id IN ({placeholders})
            """, batch).fetchall())
    finally:
        if own_connection:
            connection.close()

    thumbnails = {}
    errors = {}
    found_ids = set()
    for row in rows:
        face_id = row["face_id"]
        found_ids.add(face_id)
        destination = output_dir / f"face_{face_id}_v2.jpg"
        try:
            if not destination.is_file():
                with Image.open(row["photo_path"]) as original:
                    image = ImageOps.exif_transpose(original)
                    x = float(row["bbox_x"])
                    y = float(row["bbox_y"])
                    width = float(row["bbox_w"])
                    height = float(row["bbox_h"])
                    box = (
                        max(0, math.floor(x)),
                        max(0, math.floor(y)),
                        min(image.width, math.ceil(x + width)),
                        min(image.height, math.ceil(y + height)),
                    )
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError("face bounding box is outside the photo")
                    face = image.crop(box).convert("RGB")
                    face.thumbnail((160, 160))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    temp = tempfile.NamedTemporaryFile(dir=output_dir, delete=False)
                    temp_path = Path(temp.name)
                    try:
                        with temp:
                            face.save(temp, "JPEG", quality=90)
                        os.replace(temp_path, destination)
                    finally:
                        temp_path.unlink(missing_ok=True)
            thumbnails[face_id] = destination
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            errors[face_id] = str(exc)
            print(f"Could not create thumbnail for face {face_id}: {exc}")

    for face_id in set(face_ids) - found_ids:
        errors[face_id] = "Face or source photo is missing from the index"
        print(f"Could not create thumbnail for face {face_id}: {errors[face_id]}")
    return thumbnails, errors
