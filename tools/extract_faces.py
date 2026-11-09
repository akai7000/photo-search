import sqlite3
import sys
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.face_thumbnails import ensure_face_thumbnails
from core.database import DB_PATH


def extract_faces():
    db_path = DB_PATH
    output_dir = DB_PATH.parent / "face_thumbnails"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        face_ids = [row["id"] for row in conn.execute("SELECT id FROM faces")]
        print(f"Found {len(face_ids)} faces. Extracting...")
        thumbnails, errors = ensure_face_thumbnails(
            face_ids, connection=conn, output_dir=output_dir
        )

    print(f"Created or found {len(thumbnails)} face thumbnails in {output_dir.resolve()}.")
    if errors:
        print(f"Could not create {len(errors)} face thumbnails.")
    return len(errors)


if __name__ == "__main__":
    sys.exit(1 if extract_faces() else 0)
