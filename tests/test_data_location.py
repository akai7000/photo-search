import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from core import data_location, database, license_client


class DataLocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        patcher = patch.object(data_location, "CONFIG_PATH", self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_and_persisted_location(self):
        self.assertEqual(data_location.load_data_directory(), data_location.DEFAULT_DATA_DIR)
        data_location.save_data_directory(self.root)
        self.assertEqual(data_location.load_data_directory(), self.root)
        self.assertEqual(json.loads(self.config.read_text(encoding="utf-8"))["data_directory"], str(self.root))

    def test_download_url_setting_can_be_used_without_a_data_directory_and_is_preserved(self):
        self.config.write_text(json.dumps({"download_url": "https://example.org/downloads"}), encoding="utf-8")
        self.assertEqual(data_location.load_data_directory(), data_location.DEFAULT_DATA_DIR)
        data_location.save_data_directory(self.root)
        self.assertEqual(json.loads(self.config.read_text(encoding="utf-8")),
                         {"data_directory": str(self.root), "download_url": "https://example.org/downloads"})

    def test_license_configuration_is_persistent_and_preserves_other_configuration(self):
        self.assertFalse(license_client.is_licensed())
        self.config.write_text(json.dumps({"download_url": "https://example.org/downloads"}), encoding="utf-8")
        data_location.save_config_values({
            "license_key": "AAAA-BBBB-CCCC-DDDD",
            "license_machine_id": "d4c393eb-67cd-426f-97fb-4c2641052be8",
            "license_last_validated": datetime.now(timezone.utc).isoformat(),
            "license_expires_at": None,
            "license_permanent": True,
        })
        self.assertTrue(license_client.is_licensed())
        self.assertEqual(data_location.load_data_directory(), data_location.DEFAULT_DATA_DIR)
        data_location.save_data_directory(self.root)
        self.assertTrue(license_client.is_licensed())
        self.assertEqual(data_location.load_config()["download_url"], "https://example.org/downloads")

    def test_legacy_mock_key_does_not_unlock_indexing(self):
        self.config.write_text(
            json.dumps({"mock_license_key": "PHOTO-SEARCH-MOCK-UNLOCK"}),
            encoding="utf-8",
        )
        self.assertFalse(license_client.is_licensed())
        self.assertEqual(data_location.load_data_directory(), data_location.DEFAULT_DATA_DIR)

    def test_installed_storage_uses_local_app_data(self):
        with patch.object(sys, "frozen", True, create=True), \
             patch.dict(os.environ, {"LOCALAPPDATA": str(self.root)}):
            data, config = data_location.storage_paths()
        self.assertEqual(data, self.root / "Photo Search" / "data")
        self.assertEqual(config, self.root / "Photo Search" / "config.json")

    def test_installed_storage_requires_local_app_data(self):
        with patch.object(sys, "frozen", True, create=True), \
             patch.dict(os.environ, {"LOCALAPPDATA": ""}):
            with self.assertRaisesRegex(RuntimeError, "LOCALAPPDATA"):
                data_location.storage_paths()

    def test_saving_creates_config_directory(self):
        with patch.object(data_location, "CONFIG_PATH", self.root / "new" / "config.json"):
            data_location.save_data_directory(self.root)
            self.assertEqual(data_location.load_data_directory(), self.root)

    def test_missing_or_invalid_config_is_reported(self):
        self.config.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            data_location.load_data_directory()
        self.config.write_text('{"data_directory": "relative"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            data_location.load_data_directory()
        self.config.write_text(json.dumps({"data_directory": str(self.root / "missing")}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            data_location.load_data_directory()

    def test_prepare_new_and_existing_database(self):
        database.prepare_database(self.root)
        path = self.root / "photo_search.db"
        with patch.object(database, "DB_PATH", path):
            database.add_folder("photos")
            database.save_setting("face_page_size", 40)
        database.prepare_database(self.root)
        with patch.object(database, "DB_PATH", path):
            self.assertEqual(database.get_folders(), ["photos"])
            self.assertEqual(database.load_setting("face_page_size"), 40)

    def test_existing_photo_database_migrates_face_method_data(self):
        path = self.root / "legacy.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("""
                CREATE TABLE photos (
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
                    clip_embedding BLOB,
                    face_backend TEXT NOT NULL DEFAULT 'insightface'
                )
            """)
            conn.execute("""
                CREATE TABLE people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo_id INTEGER NOT NULL,
                    person_id INTEGER,
                    cluster_id INTEGER,
                    bbox_x REAL,
                    bbox_y REAL,
                    bbox_w REAL,
                    bbox_h REAL,
                    embedding BLOB
                )
            """)
            conn.execute("""
                CREATE TABLE folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    indexed INTEGER DEFAULT 0,
                    indexed_at TEXT,
                    total_files INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)
            """)
            conn.execute(
                "INSERT INTO photos (path, modified_time) VALUES ('current.jpg', 10)"
            )
            conn.execute(
                "INSERT INTO photos (path, modified_time, face_backend) "
                "VALUES ('legacy.jpg', 20, 'alternate')"
            )
            conn.execute("INSERT INTO people (name) VALUES ('Legacy name')")
            conn.execute("INSERT INTO folders (path, indexed) VALUES ('photos', 1)")
            conn.execute("INSERT INTO settings (key, value) VALUES ('face_backend', 'alternate')")
            conn.execute("INSERT INTO faces (photo_id, person_id, embedding) VALUES (1, 1, X'01')")
            conn.execute("INSERT INTO faces (photo_id, person_id, embedding) VALUES (2, 1, X'02')")
            conn.commit()
        database.init_db(path)
        database.init_db(path)
        with closing(database.get_connection(path)) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(photos)")}
            photos = conn.execute("SELECT modified_time FROM photos ORDER BY id").fetchall()
            faces = conn.execute("SELECT photo_id FROM faces").fetchall()
            folder = conn.execute("SELECT indexed FROM folders WHERE path = 'photos'").fetchone()
            setting = conn.execute(
                "SELECT value FROM settings WHERE key = 'face_backend'"
            ).fetchone()
        self.assertNotIn("face_backend", columns)
        self.assertEqual([row["modified_time"] for row in photos], [None, None])
        self.assertEqual(faces, [])
        self.assertEqual(folder["indexed"], 0)
        self.assertIsNone(setting)

    def test_unknown_face_model_is_rejected_without_deleting_faces(self):
        path = self.root / "version.db"
        database.init_db(path)
        with closing(database.get_connection(path)) as conn:
            conn.execute("INSERT INTO photos (path) VALUES ('photo.jpg')")
            conn.execute("INSERT INTO faces (photo_id, embedding) VALUES (1, X'01')")
            conn.execute("UPDATE index_metadata SET value = 'future-model' WHERE key = 'face_embedding_version'")
            conn.commit()
        with self.assertRaisesRegex(ValueError, "Unsupported face embedding version"):
            database.init_db(path)
        with closing(database.get_connection(path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0], 1)

    def test_existing_sface_index_is_not_migrated_again(self):
        path = self.root / "current.db"
        database.init_db(path)
        with closing(database.get_connection(path)) as conn:
            conn.execute("INSERT INTO photos (path, modified_time) VALUES ('photo.jpg', 10)")
            conn.execute("INSERT INTO faces (photo_id, embedding) VALUES (1, X'01')")
            conn.commit()
        database.init_db(path)
        with closing(database.get_connection(path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT modified_time FROM photos").fetchone()[0], 10)

    def test_unavailable_selected_directory_is_not_recreated(self):
        missing = self.root / "missing" / "photo_search.db"
        with patch.object(database, "DB_PATH", missing):
            with self.assertRaisesRegex(FileNotFoundError, "unavailable"):
                database.get_connection()
        self.assertFalse(missing.parent.exists())

    def test_corrupted_and_unrelated_databases_are_untouched(self):
        path = self.root / "photo_search.db"
        for contents in (b"not a database", b""):
            path.write_bytes(contents)
            with self.assertRaises((ValueError, sqlite3.Error)):
                database.prepare_database(self.root)
            self.assertEqual(path.read_bytes(), contents)
        path.unlink()
        with closing(sqlite3.connect(path)) as conn:
            with conn:
                conn.execute("CREATE TABLE unrelated (id INTEGER)")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "not a Photo Search index"):
            database.prepare_database(self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_existing_thumbnail_paths_follow_selected_data_directory(self):
        path = self.root / "photo_search.db"
        database.prepare_database(self.root)
        with patch.object(database, "DB_PATH", path):
            photo = database.insert_or_update_photo({
                "path": "original.jpg", "thumbnail_path": "data/thumbnails/old.jpg",
            })
            self.assertEqual(database.get_search_photo_details([photo])[photo][1],
                             str(self.root / "thumbnails" / "old.jpg"))
            database.insert_face(photo, [0, 0, 10, 10], b"")
            conn = database.get_connection()
            try:
                face_id = conn.execute("SELECT id FROM faces").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(database.get_face_photo_details([face_id])[face_id][1],
                             str(self.root / "thumbnails" / "old.jpg"))


if __name__ == "__main__":
    unittest.main()
