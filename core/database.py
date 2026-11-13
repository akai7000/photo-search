import os, sqlite3, json
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from core.data_location import DEFAULT_DATA_DIR, load_data_directory

# Database path
DB_PATH = load_data_directory() / "photo_search.db"

def get_connection(path: Path | None = None):
    """Return a connection to the SQLite database."""
    path = path if path is not None else DB_PATH
    if path.parent == DEFAULT_DATA_DIR:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.is_dir():
        raise FileNotFoundError(f"Data directory is unavailable: {path.parent}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # Makes rows accessible by column name
    return conn

def init_db(path: Path | None = None):
    """Create tables if they don't exist."""
    conn = get_connection(path)
    try:
        _init_db(conn)
    finally:
        conn.close()


def _init_db(conn):
    cursor = conn.cursor()

    # Photos table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            file_hash TEXT,
            file_size INTEGER,
            modified_time REAL,
            date_taken TEXT,
            width INTEGER,
            height INTEGER,
            thumbnail_path TEXT,
            indexed_at TEXT,
            clip_embedding BLOB
        )
    """)
    photo_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(photos)")}
    remove_legacy_backend_column = "face_backend" in photo_columns

    # People table (named clusters)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TEXT
        )
    """)

    # Faces table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            person_id INTEGER,
            cluster_id INTEGER,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            embedding BLOB,
            FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE SET NULL
        )
    """)

    columns = {row["name"] for row in cursor.execute("PRAGMA table_info(faces)")}
    if "cluster_id" not in columns:
        cursor.execute("BEGIN")
        cursor.execute("ALTER TABLE faces ADD COLUMN cluster_id INTEGER")
        cursor.execute("""
            UPDATE faces SET cluster_id = person_id, person_id = NULL
            WHERE person_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM people WHERE people.id = faces.person_id)
        """)

    # Settings table - deprecated
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Folders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            indexed INTEGER DEFAULT 0,
            indexed_at TEXT,
            total_files INTEGER
        )
    """)
    folder_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(folders)")}
    if "total_files" not in folder_columns:
        cursor.execute("ALTER TABLE folders ADD COLUMN total_files INTEGER")

    if remove_legacy_backend_column:
        legacy_photos = cursor.execute(
            "SELECT COUNT(*) FROM photos WHERE face_backend IS NOT 'insightface'"
        ).fetchone()[0]
        if legacy_photos:
            cursor.execute(
                "UPDATE photos SET modified_time = NULL "
                "WHERE face_backend IS NOT 'insightface'"
            )
            cursor.execute(
                "DELETE FROM faces WHERE photo_id IN "
                "(SELECT id FROM photos WHERE face_backend IS NOT 'insightface')"
            )
            cursor.execute(
                "DELETE FROM people WHERE NOT EXISTS "
                "(SELECT 1 FROM faces WHERE faces.person_id = people.id)"
            )
            cursor.execute("UPDATE folders SET indexed = 0")
        cursor.execute("ALTER TABLE photos DROP COLUMN face_backend")

    cursor.execute("DELETE FROM settings WHERE key = 'face_backend'")

    # Indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_hash ON photos(file_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")

    conn.commit()
    print("Database initialized.")


def prepare_database(directory: Path) -> None:
    """Validate an existing index or initialize a new one without selecting it."""
    if not directory.is_dir():
        raise ValueError(f"Data directory does not exist: {directory}")
    path = directory / "photo_search.db"
    if path.exists():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Database is empty or not a file: {path}")
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            result = conn.execute("PRAGMA quick_check").fetchall()
            if result != [("ok",)]:
                raise ValueError(f"Database integrity check failed: {result}")
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )}
            if not {"photos", "faces", "people", "folders"}.issubset(tables):
                raise ValueError("The selected database is not a Photo Search index")
        finally:
            conn.close()
    init_db(path)


def resolve_thumbnail_path(path: str | None) -> str | None:
    if path is None:
        return None
    thumbnail = Path(path)
    if thumbnail.parent.name == "thumbnails":
        return str(DB_PATH.parent / "thumbnails" / thumbnail.name)
    return path

# --- Helper functions ---

def photo_exists(path: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM photos WHERE path = ?", (path,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def insert_or_update_photo(data: Dict) -> int:
    """Insert or update a photo. Returns the photo id."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO photos (
            path, file_hash, file_size, modified_time,
            date_taken, width, height, thumbnail_path, indexed_at, clip_embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            file_hash = excluded.file_hash,
            file_size = excluded.file_size,
            modified_time = excluded.modified_time,
            date_taken = excluded.date_taken,
            width = excluded.width,
            height = excluded.height,
            thumbnail_path = excluded.thumbnail_path,
            indexed_at = excluded.indexed_at,
            clip_embedding = excluded.clip_embedding
    """, (
        data["path"],
        data.get("file_hash"),
        data.get("file_size"),
        data.get("modified_time"),
        data.get("date_taken"),
        data.get("width"),
        data.get("height"),
        data.get("thumbnail_path"),
        datetime.utcnow().isoformat(),
        data.get("clip_embedding"),
    ))

    # Get the id
    cursor.execute("SELECT id FROM photos WHERE path = ?", (data["path"],))
    photo_id = cursor.fetchone()["id"]

    conn.commit()
    conn.close()
    return photo_id


def delete_faces_for_photo(photo_id: int) -> None:
    """Replace all face detections for one photo after a successful detection pass."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
        conn.execute(
            "DELETE FROM people WHERE NOT EXISTS "
            "(SELECT 1 FROM faces WHERE faces.person_id = people.id)"
        )
        conn.commit()
    finally:
        conn.close()


def get_all_photos() -> List[sqlite3.Row]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos ORDER BY date_taken DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_photo_count() -> int:
    """Return the total number of photos in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM photos")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_photo_ids_by_date(start: date | None, end: date | None) -> list[int]:
    """Return photos in the half-open capture-date interval [start, end)."""
    if start is None and end is None:
        raise ValueError("At least one capture-date bound is required")
    clauses = ["date_taken IS NOT NULL"]
    params = []
    if start is not None:
        clauses.append("date(date_taken) >= ?")
        params.append(start.isoformat())
    if end is not None:
        clauses.append("date(date_taken) < ?")
        params.append(end.isoformat())

    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT id FROM photos WHERE {' AND '.join(clauses)} "
            "ORDER BY date_taken DESC, id",
            params,
        ).fetchall()
        return [row["id"] for row in rows]
    finally:
        conn.close()

def delete_all_photos():
    """Delete everything from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faces")
    cursor.execute("DELETE FROM photos")
    cursor.execute("DELETE FROM people")
    cursor.execute("DELETE FROM settings")
    cursor.execute("UPDATE folders SET indexed = 0, indexed_at = NULL, total_files = NULL")
    conn.commit()
    conn.close()

def save_setting(key: str, value: Any):
    """Save a setting. Value is automatically JSON-encoded."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, json.dumps(value))
    )
    conn.commit()
    conn.close()

def load_setting(key: str, default=None) -> Any:
    """Load a setting. Returns default if the key doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default

def mark_folder_indexed(folder: str):
    """Mark a folder as indexed with the current timestamp."""
    folders_status = load_setting("folders_status", {})
    folders_status[folder] = datetime.utcnow().isoformat()
    save_setting("folders_status", folders_status)

def get_folder_status(folder: str) -> str:
    """Return a human-readable status for a folder."""
    folders_status = load_setting("folders_status", {})
    if folder in folders_status:
        return "Indexed"
    return "Not indexed"

def insert_face(photo_id: int, bbox: list, embedding: bytes, det_score: float = 0.0):
    """Insert a detected face linked to a photo."""
    conn = get_connection()
    cursor = conn.cursor()

    x1, y1, x2, y2 = bbox
    cursor.execute("""
        INSERT INTO faces (
            photo_id, bbox_x, bbox_y, bbox_w, bbox_h, embedding
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        photo_id,
        float(x1),
        float(y1),
        float(x2 - x1),
        float(y2 - y1),
        embedding,
    ))

    conn.commit()
    conn.close()

def get_face_photo_details(face_ids: list[int]) -> dict[int, tuple[str, str | None]]:
    """Return original and cached preview paths for the requested faces."""
    if not face_ids:
        return {}
    conn = get_connection()
    try:
        paths = {}
        for offset in range(0, len(face_ids), 500):
            batch = face_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT f.id, p.path, p.thumbnail_path FROM faces f JOIN photos p ON p.id = f.photo_id "
                f"WHERE f.id IN ({placeholders})",
                batch,
            )
            paths.update((row["id"], (row["path"], resolve_thumbnail_path(row["thumbnail_path"]))) for row in rows)
        return paths
    finally:
        conn.close()


def get_clusters():
    """Return named and unnamed groups with their face IDs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT cluster_id, person_id, person_name, face_count, face_ids FROM (
            SELECT cluster_id, NULL AS person_id, NULL AS person_name,
                  COUNT(*) AS face_count, GROUP_CONCAT(id) AS face_ids,
                  0 AS is_named, MAX(id) AS newest
            FROM faces WHERE cluster_id IS NOT NULL AND person_id IS NULL
            GROUP BY cluster_id
            UNION ALL
            SELECT NULL AS cluster_id, people.id AS person_id, people.name AS person_name,
                  COUNT(*) AS face_count, GROUP_CONCAT(faces.id) AS face_ids,
                  1 AS is_named, MAX(faces.id) AS newest
            FROM faces JOIN people ON people.id = faces.person_id
            GROUP BY people.id, people.name
        )
        ORDER BY is_named, newest DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def suggest_group_names(threshold: float = 0.45) -> dict[int, tuple[str, float]]:
    """Suggest a saved name for each unnamed group by comparing average face embeddings.

    Returns {cluster_id: (name, cosine_similarity)} for groups that match a named person.
    """
    import numpy as np

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT cluster_id, person_id, embedding FROM faces "
            "WHERE embedding IS NOT NULL AND (person_id IS NOT NULL OR cluster_id IS NOT NULL)"
        ).fetchall()
        names = {row["id"]: row["name"] for row in conn.execute("SELECT id, name FROM people")}
    finally:
        conn.close()

    people: dict[int, list] = {}
    groups: dict[int, list] = {}
    for row in rows:
        blob = row["embedding"]
        if len(blob) % 4:
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if row["person_id"] is not None:
            people.setdefault(row["person_id"], []).append(vector)
        else:
            groups.setdefault(row["cluster_id"], []).append(vector)

    def centroid(vectors):
        vectors = [v for v in vectors if v.shape == vectors[0].shape]
        mean = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm else None

    person_centroids = {pid: c for pid, vs in people.items() if (c := centroid(vs)) is not None}
    if not person_centroids:
        return {}
    person_ids = list(person_centroids)
    dimension = person_centroids[person_ids[0]].shape
    person_ids = [pid for pid in person_ids if person_centroids[pid].shape == dimension]
    matrix = np.stack([person_centroids[pid] for pid in person_ids])

    suggestions = {}
    for cluster_id, vectors in groups.items():
        group_centroid = centroid(vectors)
        if group_centroid is None or group_centroid.shape != matrix.shape[1:]:
            continue
        scores = matrix @ group_centroid
        best = int(np.argmax(scores))
        if scores[best] >= threshold:
            suggestions[cluster_id] = (names[person_ids[best]], float(scores[best]))
    return suggestions


def get_people_names() -> list[str]:
    """Return saved names for the person selector."""
    conn = get_connection()
    try:
        names = [row["name"] for row in conn.execute("SELECT name FROM people")]
        return sorted(names, key=str.casefold)
    finally:
        conn.close()


def get_or_create_person(name: str) -> int:
    """Get existing person ID or create a new one."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM people WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        person_id = row["id"]
    else:
        cursor.execute(
            "INSERT INTO people (name, created_at) VALUES (?, ?)",
            (name, datetime.utcnow().isoformat())
        )
        person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return person_id

def _find_or_create_person(conn: sqlite3.Connection, name: str) -> int:
    people = conn.execute("SELECT id, name FROM people").fetchall()
    person = next((row for row in people if row["name"].casefold() == name.casefold()), None)
    if person is not None:
        return person["id"]
    cursor = conn.execute(
        "INSERT INTO people (name, created_at) VALUES (?, ?)",
        (name, datetime.utcnow().isoformat()),
    )
    return cursor.lastrowid


def assign_name_to_cluster(cluster_id: int, name: str):
    """Assign a name to all faces in a cluster."""
    name = name.strip()
    if not name:
        raise ValueError("A name is required")
    conn = get_connection()
    try:
        with conn:
            if not conn.execute(
                "SELECT 1 FROM faces WHERE cluster_id = ? AND person_id IS NULL",
                (cluster_id,),
            ).fetchone():
                raise ValueError(f"Cluster {cluster_id} no longer exists")
            person_id = _find_or_create_person(conn, name)
            conn.execute(
                "UPDATE faces SET person_id = ?, cluster_id = NULL "
                "WHERE cluster_id = ? AND person_id IS NULL",
                (person_id, cluster_id),
            )
    finally:
        conn.close()


def _group_face_ids(conn: sqlite3.Connection, cluster_id: int | None, person_id: int | None) -> set[int]:
    if (cluster_id is None) == (person_id is None):
        raise ValueError("Specify exactly one face group")
    if person_id is not None:
        rows = conn.execute("SELECT id FROM faces WHERE person_id = ?", (person_id,))
    else:
        rows = conn.execute(
            "SELECT id FROM faces WHERE cluster_id = ? AND person_id IS NULL", (cluster_id,)
        )
    return {row["id"] for row in rows}


def _remove_empty_person(conn: sqlite3.Connection, person_id: int | None):
    if person_id is not None:
        conn.execute(
            "DELETE FROM people WHERE id = ? AND NOT EXISTS "
            "(SELECT 1 FROM faces WHERE person_id = ?)",
            (person_id, person_id),
        )


def assign_selected_faces(
    face_ids: list[int], name: str, *, cluster_id: int | None, person_id: int | None
):
    """Move selected faces from one group to a named person."""
    name = name.strip()
    selected = set(face_ids)
    if not name or not selected:
        raise ValueError("Select at least one face and enter a name")
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            if not selected <= _group_face_ids(conn, cluster_id, person_id):
                raise ValueError("The face group changed; reopen Name People and try again")
            target_id = _find_or_create_person(conn, name)
            conn.executemany(
                "UPDATE faces SET person_id = ?, cluster_id = NULL WHERE id = ?",
                [(target_id, face_id) for face_id in selected],
            )
            _remove_empty_person(conn, person_id)
    finally:
        conn.close()


def _delete_faces(
    face_ids: list[int], *, cluster_id: int | None, person_id: int | None, whole_group: bool
) -> list[int]:
    expected = set(face_ids)
    if not expected:
        raise ValueError("Select at least one face")
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            current = _group_face_ids(conn, cluster_id, person_id)
            if not expected <= current or (whole_group and expected != current):
                raise ValueError("The face group changed; reopen Name People and try again")
            conn.executemany("DELETE FROM faces WHERE id = ?", [(face_id,) for face_id in expected])
            _remove_empty_person(conn, person_id)
    finally:
        conn.close()
    return sorted(expected)


def delete_face_group(
    face_ids: list[int], *, cluster_id: int | None, person_id: int | None
) -> list[int]:
    """Delete exactly the face detections shown in a group, not their photos."""
    return _delete_faces(face_ids, cluster_id=cluster_id, person_id=person_id, whole_group=True)


def delete_selected_faces(
    face_ids: list[int], *, cluster_id: int | None, person_id: int | None
) -> list[int]:
    """Delete selected detections from a group, not their photos."""
    return _delete_faces(face_ids, cluster_id=cluster_id, person_id=person_id, whole_group=False)


def rename_person(person_id: int, name: str):
    """Rename a person, merging their faces with an existing matching name."""
    name = name.strip()
    if not name:
        raise ValueError("A name is required")
    conn = get_connection()
    try:
        with conn:
            people = conn.execute("SELECT id, name FROM people").fetchall()
            if not any(row["id"] == person_id for row in people):
                raise ValueError(f"Person {person_id} no longer exists")
            other = next(
                (row for row in people
                 if row["id"] != person_id and row["name"].casefold() == name.casefold()),
                None,
            )
            if other is None:
                conn.execute("UPDATE people SET name = ? WHERE id = ?", (name, person_id))
            else:
                conn.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (other["id"], person_id))
                conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
    finally:
        conn.close()


GROUP_WORDS = {"group", "groups", "crowd", "crowds", "together"}
FILLER_WORDS = {
    "photo", "photos", "picture", "pictures", "pic", "pics", "image", "images",
    "including", "include", "includes", "with", "of", "and", "a", "an", "the",
    "containing", "featuring", "in", "that", "has", "have", "where", "is", "are",
    "showing", "show", "me", "find", "all", "any", "some", "having", "also",
}
MIN_GROUP_FACES = 3
ONLY_WORDS = {"only", "just", "alone", "exclusively"}
NEGATION_WORDS = {"without", "excluding", "exclude", "except", "not", "no", "minus", "besides"}


def search_people_query(query: str) -> dict | None:
    """Find saved names or individual name parts in a free-text query.

    Returns None when no saved name appears in the query. Otherwise returns a dict with
    photo_ids (photos containing every named person, and enough faces for a "group"
    request), names, and remaining_text that still needs a visual (CLIP) search.
    """
    import re

    conn = get_connection()
    try:
        people = conn.execute("SELECT id, name FROM people").fetchall()
        by_name: dict[str, list[int]] = {}
        for row in people:
            key = row["name"].strip().casefold()
            if key:
                by_name.setdefault(key, []).append(row["id"])
        by_term = {name: set(ids) for name, ids in by_name.items()}
        reserved = GROUP_WORDS | FILLER_WORDS | ONLY_WORDS | NEGATION_WORDS
        for name, ids in by_name.items():
            parts = name.split()
            if len(parts) > 1:
                for part in parts:
                    if part not in reserved:
                        by_term.setdefault(part, set()).update(ids)

        matched: list[list[int]] = []
        names: list[str] = []
        if not by_term:
            return None
        pattern = r"(?<!\w)(?:" + "|".join(
            re.escape(term) for term in sorted(by_term, key=len, reverse=True)
        ) + r")(?!\w)"

        def replace_name(match):
            term = match.group()
            token = f" zzperson{len(matched)} "
            matched.append(sorted(by_term[term]))
            names.append(term)
            return token

        remaining = re.sub(pattern, replace_name, query.casefold())
        if not matched:
            return None

        words = re.findall(r"\w+", remaining)
        is_group = any(word in GROUP_WORDS for word in words)
        only = any(word in ONLY_WORDS for word in words)

        # A name is excluded when a negation word precedes it, skipping filler and earlier excluded names.
        excluded: set[int] = set()
        for pos, word in enumerate(words):
            if not word.startswith("zzperson"):
                continue
            back = pos - 1
            while back >= 0 and (
                words[back] in FILLER_WORDS
                or (words[back].startswith("zzperson") and int(words[back][8:]) in excluded)
            ):
                back -= 1
            if back >= 0 and words[back] in NEGATION_WORDS:
                excluded.add(int(word[8:]))

        leftover = [
            w for w in words
            if not w.startswith("zzperson") and w not in GROUP_WORDS and w not in FILLER_WORDS
            and w not in ONLY_WORDS and w not in NEGATION_WORDS
        ]

        def photos_with(ids):
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT DISTINCT photo_id FROM faces WHERE person_id IN ({placeholders})", ids
            ).fetchall()
            return {row["photo_id"] for row in rows}

        included = [ids for i, ids in enumerate(matched) if i not in excluded]
        excluded_ids = [pid for i, ids in enumerate(matched) if i in excluded for pid in ids]
        included_ids = {pid for ids in included for pid in ids}
        if included:
            photo_ids = set.intersection(*(photos_with(ids) for ids in included))
        else:
            photo_ids = {row["photo_id"] for row in conn.execute("SELECT DISTINCT photo_id FROM faces")}
        if excluded_ids:
            photo_ids -= photos_with(excluded_ids)

        face_counts = {
            row["photo_id"]: row["n"]
            for row in conn.execute("SELECT photo_id, COUNT(*) AS n FROM faces GROUP BY photo_id")
        }
        if only and included_ids:
            # Every detected face (named or not) must belong to the requested people.
            placeholders = ",".join("?" for _ in included_ids)
            strangers = {
                row["photo_id"] for row in conn.execute(
                    f"SELECT DISTINCT photo_id FROM faces "
                    f"WHERE person_id IS NULL OR person_id NOT IN ({placeholders})",
                    list(included_ids),
                )
            }
            photo_ids -= strangers
        if is_group:
            photo_ids = {p for p in photo_ids if face_counts.get(p, 0) >= MIN_GROUP_FACES}
        ordered = sorted(photo_ids, key=lambda p: (-face_counts.get(p, 0), p))
        return {
            "photo_ids": ordered,
            "names": names,
            "remaining_text": " ".join(leftover),
            "is_group": is_group,
        }
    finally:
        conn.close()


def find_photos_by_person_name(name: str) -> list[int] | None:
    """Return distinct photos for an exact, case-insensitive name, or None if unknown."""
    normalized = name.strip().casefold()
    conn = get_connection()
    try:
        people = conn.execute("SELECT id, name FROM people").fetchall()
        person_ids = [row["id"] for row in people if row["name"].strip().casefold() == normalized]
        if not person_ids:
            return None
        placeholders = ",".join("?" for _ in person_ids)
        rows = conn.execute(
            f"SELECT DISTINCT photo_id FROM faces WHERE person_id IN ({placeholders}) ORDER BY photo_id",
            person_ids,
        ).fetchall()
        return [row["photo_id"] for row in rows]
    finally:
        conn.close()

def add_folder(path: str):
    """Add a folder if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO folders (path) VALUES (?)",
        (path,)
    )
    conn.commit()
    conn.close()

def remove_folder_db(path: str):
    """Remove a folder from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM folders WHERE path = ?", (path,))
    conn.commit()
    conn.close()

def get_folders() -> list[str]:
    """Return the list of folder paths."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM folders ORDER BY path")
    rows = cursor.fetchall()
    conn.close()
    return [row["path"] for row in rows]

def mark_folder_indexed(path: str, *, update_timestamp: bool = True, mark_complete: bool = True):
    """Record a completed scan and optionally the time of its latest changes."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE folders SET indexed = CASE WHEN ? THEN 1 ELSE indexed END, "
        "indexed_at = CASE WHEN ? THEN ? ELSE indexed_at END WHERE path = ?",
        (mark_complete, update_timestamp, datetime.utcnow().isoformat(), path)
    )
    conn.commit()
    conn.close()

def set_folder_total(path: str, total: int):
    """Record the scan total and mark the new scan incomplete until it finishes."""
    conn = get_connection()
    try:
        conn.execute("UPDATE folders SET total_files = ?, indexed = 0 WHERE path = ?", (total, path))
        conn.commit()
    finally:
        conn.close()


def get_folder_progress(path: str) -> dict:
    """Return indexing state, file counts and the last change timestamp."""
    prefix = str(Path(path)).rstrip("\\/") + os.sep
    prefix_end = prefix[:-1] + chr(ord(prefix[-1]) + 1)
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT indexed, total_files, indexed_at FROM folders WHERE path = ?", (path,)
        ).fetchone()
        indexed = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE path >= ? AND path < ?", (prefix, prefix_end)
        ).fetchone()[0]
    finally:
        conn.close()
    total = row["total_files"] if row else None
    if indexed == 0:
        state = "complete" if total == 0 and row and row["indexed"] else "not"
    elif total is None:
        # Folders indexed before totals were recorded.
        state = "complete" if row and row["indexed"] else "partial"
    else:
        state = "complete" if indexed >= total else "partial"
    return {"state": state, "indexed": indexed, "total": total if total is not None else indexed,
            "indexed_at": row["indexed_at"] if row else None,
            "scan_completed": bool(row and row["indexed"])}


def get_search_photo_details(photo_ids: list[int]) -> dict[int, tuple[str, str | None, str | None]]:
    """Fetch result metadata in batches without opening a connection per result."""
    conn = get_connection()
    try:
        details = {}
        for offset in range(0, len(photo_ids), 500):
            batch = photo_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT id, path, thumbnail_path, date_taken FROM photos WHERE id IN ({placeholders})",
                batch,
            )
            details.update((row["id"], (row["path"], resolve_thumbnail_path(row["thumbnail_path"]), row["date_taken"])) for row in rows)
        return details
    finally:
        conn.close()


def format_folder_status(progress: dict) -> str:
    """Format a folder progress record for display."""
    if progress["state"] == "complete":
        return "Indexed"
    if progress["state"] == "partial":
        return f"Partially indexed ({progress['indexed']} of {progress['total']})"
    return "Not indexed"


def get_folder_status(path: str) -> str:
    """Return the indexing status for a folder."""
    return format_folder_status(get_folder_progress(path))