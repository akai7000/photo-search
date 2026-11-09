import os
from pathlib import Path

from core import database, face_thumbnails


def _normalized(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _inside(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath((path, folder)) == folder and path != folder
    except ValueError:
        return False


def remove_folder_and_index(folder: str) -> tuple[int, list[str]]:
    """Remove unshared index records atomically, then remove their generated images."""
    conn = database.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            folders = [row["path"] for row in conn.execute("SELECT path FROM folders")]
            if folder not in folders:
                raise ValueError("The folder is no longer in the searchable folders list.")
            removed_root = _normalized(folder)
            remaining_roots = [_normalized(path) for path in folders if path != folder]
            photos = conn.execute("SELECT id, path, thumbnail_path FROM photos").fetchall()
            removed = []
            kept = []
            for photo in photos:
                path = _normalized(photo["path"])
                if _inside(path, removed_root) and not any(_inside(path, root) for root in remaining_roots):
                    removed.append(photo)
                else:
                    kept.append(photo)

            face_ids = []
            for offset in range(0, len(removed), 500):
                batch = removed[offset:offset + 500]
                ids = [row["id"] for row in batch]
                placeholders = ",".join("?" for _ in ids)
                face_ids.extend(row["id"] for row in conn.execute(
                    f"SELECT id FROM faces WHERE photo_id IN ({placeholders})", ids
                ))
                conn.execute(f"DELETE FROM faces WHERE photo_id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", ids)
            conn.execute("DELETE FROM folders WHERE path = ?", (folder,))
    finally:
        conn.close()

    kept_thumbnails = {
        _normalized(database.resolve_thumbnail_path(photo["thumbnail_path"]))
        for photo in kept if photo["thumbnail_path"]
    }
    original_paths = {_normalized(photo["path"]) for photo in photos}
    thumbnail_dir = (database.DB_PATH.parent / "thumbnails").resolve()
    errors = face_thumbnails.delete_face_thumbnails(face_ids)
    for thumbnail in {database.resolve_thumbnail_path(row["thumbnail_path"]) for row in removed}:
        if not thumbnail:
            continue
        path = Path(thumbnail)
        normalized = _normalized(thumbnail)
        if normalized in kept_thumbnails:
            continue
        try:
            if normalized in original_paths or path.resolve().parent != thumbnail_dir:
                errors.append(f"Did not remove unsafe thumbnail path: {path}")
                continue
            path.unlink(missing_ok=True)
        except (OSError, RuntimeError) as exc:
            errors.append(f"Could not remove thumbnail {path}: {exc}")
    return len(removed), errors
