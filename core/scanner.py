import hashlib, exifread, time
from pathlib import Path
from typing import List, Set
from PIL import Image, ImageOps, ExifTags
from datetime import datetime
from core.database import insert_or_update_photo, get_connection, mark_folder_indexed, insert_face, set_folder_total
from core.clip_engine import get_image_embedding
from core.face_engine import detect_faces

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".raw"}
THUMBNAIL_DIR = Path("data/thumbnails")

def compute_file_hash(path: Path) -> str:
    """Compute a fast hash of the file (first 1 MB + size + mtime is often enough, but full hash is safer)."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def scan_folders(folders: List[str], progress_callback=None, cancel_event=None) -> dict:
    """
    Scan the given folders for image files and insert/update them in the database.
    Returns a summary dict.
    """
    print(f"Starting scan of {len(folders)} folder(s)...")
    added = 0
    updated = 0
    skipped = 0
    errors = 0

    all_paths: Set[str] = set()

    # Count everything first so progress can show "n of total".
    work: list[tuple[str, Path]] = []
    folder_totals: dict[str, int] = {}
    for folder_str in folders:
        folder = Path(folder_str)
        if not folder.exists():
            continue
        folder_files = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        set_folder_total(folder_str, len(folder_files))
        folder_totals[folder_str] = len(folder_files)
        work.extend((folder_str, p) for p in folder_files)

    processed_in_folder: dict[str, int] = {f: 0 for f in folder_totals}
    changed_folders: set[str] = set()
    current_root = ""

    def report(path):
        if progress_callback:
            progress_callback(
                processed_in_folder.get(current_root, 0),
                folder_totals.get(current_root, 0),
                str(path),
                current_root,
            )

    report("")

    cancelled = False
    for current_root, path in work:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        all_paths.add(str(path))

        try:
            stat = path.stat()

            # Quick check: skip if already indexed and unchanged
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_hash, modified_time, thumbnail_path FROM photos WHERE path = ?",
                (str(path),)
            )
            row = cursor.fetchone()
            conn.close()

            if row is not None and abs(row["modified_time"] - stat.st_mtime) < 1.0:
                expected_thumbnail = thumbnail_path_for(path, THUMBNAIL_DIR)
                if row["thumbnail_path"] != str(expected_thumbnail) or not expected_thumbnail.is_file():
                    thumbnail_path, _, _ = generate_thumbnail(path, THUMBNAIL_DIR)
                    if thumbnail_path is None:
                        raise RuntimeError("Could not refresh thumbnail")
                    conn = get_connection()
                    try:
                        conn.execute(
                            "UPDATE photos SET thumbnail_path = ? WHERE path = ?",
                            (thumbnail_path, str(path)),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                skipped += 1
                processed_in_folder[current_root] += 1
                report(path)
                continue

            # Expensive work only for new or changed files
            file_hash = compute_file_hash(path)
            date_taken = extract_date_taken(path)
            thumbnail_path, width, height = generate_thumbnail(path, THUMBNAIL_DIR)
            clip_embedding = get_image_embedding(path)

            data = {
                "path": str(path),
                "file_hash": file_hash,
                "file_size": stat.st_size,
                "modified_time": stat.st_mtime,
                "date_taken": date_taken,
                "width": width,
                "height": height,
                "thumbnail_path": thumbnail_path,
                "clip_embedding": clip_embedding.tobytes() if clip_embedding is not None else None,
            }

            # Insert or update photo and get its ID
            photo_id = insert_or_update_photo(data)
            changed_folders.add(current_root)

            # Face detection
            faces = detect_faces(path)
            for face in faces:
                insert_face(
                    photo_id=photo_id,
                    bbox=face["bbox"],
                    embedding=face["embedding"].tobytes(),
                    det_score=face["det_score"],
                )

            if row is None:
                added += 1
            else:
                updated += 1

        except Exception as e:
            errors += 1
            print(f"Error processing {path}: {e}")

        processed_in_folder[current_root] += 1
        report(path)

    # A cancelled scan leaves folders unmarked; their status comes from the photos indexed so far.
    for folder in folder_totals:
        if not cancelled or folder in changed_folders:
            mark_folder_indexed(folder, update_timestamp=folder in changed_folders, mark_complete=not cancelled)

    return {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
        "total_found": len(all_paths),
        "cancelled": cancelled,
    }

def extract_date_taken(path: Path) -> str | None:
    """Try to extract the date the photo was taken from EXIF."""
    try:
        with path.open("rb") as f:
            tags = exifread.process_file(f, details=False)
            if "EXIF DateTimeOriginal" in tags:
                date_str = str(tags["EXIF DateTimeOriginal"])
                # Convert "2023:10:15 14:30:22" → ISO format
                dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                return dt.isoformat()
    except Exception:
        pass
    return None

def thumbnail_path_for(path: Path, thumb_dir: Path) -> Path:
    path_hash = hashlib.sha256(str(path.absolute()).encode("utf-8")).hexdigest()
    return thumb_dir / f"{path_hash}_thumb_v2.jpg"


def generate_thumbnail(path: Path, thumb_dir: Path, size=(300, 300)) -> tuple[str | None, int | None, int | None]:
    """Generate a thumbnail and return (thumbnail_path, width, height)."""
    try:
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumbnail_path_for(path, thumb_dir)

        with Image.open(path) as img:
            width, height = img.size          # ← get original dimensions
            img = ImageOps.exif_transpose(img)
            img.thumbnail(size)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(thumb_path, "JPEG", quality=85)

        return str(thumb_path), width, height
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return None, None, None