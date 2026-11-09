import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from core import data_location, database


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

    def test_missing_or_invalid_config_is_reported(self):
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
