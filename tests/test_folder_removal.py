import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from core import database, face_thumbnails
from core.folder_removal import remove_folder_and_index


class FolderRemovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = patch.object(database, "DB_PATH", self.root / "test.db")
        self.db.start()
        self.addCleanup(self.db.stop)
        self.face_dir = self.root / "face_thumbnails"
        self.face_dir.mkdir()
        face_patch = patch.object(face_thumbnails, "FACE_THUMBNAIL_DIR", self.face_dir)
        face_patch.start()
        self.addCleanup(face_patch.stop)
        self.thumb_dir = self.root / "thumbnails"
        self.thumb_dir.mkdir()
        database.init_db()

    def add_photo(self, path, thumb_name, name="Alice"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
        thumbnail = self.thumb_dir / thumb_name
        thumbnail.write_bytes(b"thumbnail")
        photo_id = database.insert_or_update_photo({"path": str(path), "thumbnail_path": str(thumbnail)})
        database.insert_face(photo_id, [0, 0, 10, 10], b"embedding")
        with closing(database.get_connection()) as conn, conn:
            face_id = conn.execute("SELECT id FROM faces WHERE photo_id = ?", (photo_id,)).fetchone()[0]
            conn.execute("INSERT OR IGNORE INTO people (name) VALUES (?)", (name,))
            conn.execute("UPDATE faces SET person_id = (SELECT id FROM people WHERE name = ?) WHERE id = ?",
                         (name, face_id))
        for suffix in ("", "_v2"):
            (self.face_dir / f"face_{face_id}{suffix}.jpg").write_bytes(b"face")
        return photo_id, face_id, thumbnail

    def test_removal_keeps_shared_and_unrelated_photos_and_people(self):
        parent = self.root / "photos"
        child = parent / "shared"
        sibling = self.root / "photos-other"
        for folder in (parent, child, sibling):
            database.add_folder(str(folder))
        removed_path = parent / "a.jpg"
        shared_path = child / "b.jpg"
        sibling_path = sibling / "c.jpg"
        removed, face, thumb = self.add_photo(removed_path, "a.jpg", "Alice")
        shared, shared_face, shared_thumb = self.add_photo(shared_path, "b.jpg", "Bob")
        other, _, _ = self.add_photo(sibling_path, "c.jpg", "Carol")

        count, errors = remove_folder_and_index(str(parent))

        self.assertEqual((count, errors), (1, []))
        self.assertEqual(set(database.get_folders()), {str(child), str(sibling)})
        self.assertEqual(database.get_photo_count(), 2)
        self.assertEqual(database.find_photos_by_person_name("Alice"), [])
        self.assertEqual(database.find_photos_by_person_name("Bob"), [shared])
        self.assertEqual(database.find_photos_by_person_name("Carol"), [other])
        self.assertTrue(removed_path.exists())
        self.assertFalse(thumb.exists())
        self.assertTrue(shared_thumb.exists())
        for suffix in ("", "_v2"):
            self.assertFalse((self.face_dir / f"face_{face}{suffix}.jpg").exists())
            self.assertTrue((self.face_dir / f"face_{shared_face}{suffix}.jpg").exists())
        with closing(database.get_connection()) as conn:
            self.assertIsNone(conn.execute("SELECT id FROM photos WHERE id = ?", (removed,)).fetchone())
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0], 3)

    def test_removing_child_keeps_everything_when_parent_is_selected(self):
        parent = self.root / "parent"
        child = parent / "child"
        database.add_folder(str(parent))
        database.add_folder(str(child))
        path = child / "photo.jpg"
        _, face, thumbnail = self.add_photo(path, "photo.jpg")
        self.assertEqual(remove_folder_and_index(str(child)), (0, []))
        self.assertEqual(database.get_photo_count(), 1)
        self.assertTrue(thumbnail.exists())
        self.assertTrue((self.face_dir / f"face_{face}_v2.jpg").exists())

    def test_cleanup_failure_is_reported_after_database_commit(self):
        folder = self.root / "photos"
        database.add_folder(str(folder))
        path = folder / "photo.jpg"
        _, _, thumbnail = self.add_photo(path, "photo.jpg")
        original_unlink = Path.unlink
        def fail_thumbnail(target, *args, **kwargs):
            if target == thumbnail:
                raise PermissionError("locked")
            return original_unlink(target, *args, **kwargs)
        with patch.object(Path, "unlink", fail_thumbnail):
            count, errors = remove_folder_and_index(str(folder))
        self.assertEqual(count, 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("locked", errors[0])
        self.assertEqual(database.get_folders(), [])
        self.assertTrue(path.exists())

    def test_outside_thumbnail_path_is_not_deleted(self):
        folder = self.root / "photos"
        database.add_folder(str(folder))
        path = folder / "photo.jpg"
        self.add_photo(path, "photo.jpg")
        outside = self.root / "other.jpg"
        outside.write_bytes(b"not generated")
        with closing(database.get_connection()) as conn, conn:
            conn.execute("UPDATE photos SET thumbnail_path = ?", (str(outside),))
        _, errors = remove_folder_and_index(str(folder))
        self.assertTrue(outside.exists())
        self.assertIn("unsafe thumbnail", errors[0])

    def test_shared_thumbnail_is_not_deleted(self):
        first = self.root / "first"
        second = self.root / "second"
        database.add_folder(str(first))
        database.add_folder(str(second))
        _, _, thumbnail = self.add_photo(first / "a.jpg", "shared.jpg")
        other, _, _ = self.add_photo(second / "b.jpg", "other.jpg")
        with closing(database.get_connection()) as conn, conn:
            conn.execute("UPDATE photos SET thumbnail_path = ? WHERE id = ?", (str(thumbnail), other))
        self.assertEqual(remove_folder_and_index(str(first)), (1, []))
        self.assertTrue(thumbnail.exists())
        self.assertEqual(database.get_photo_count(), 1)

    def test_face_cleanup_failure_is_reported(self):
        folder = self.root / "photos"
        database.add_folder(str(folder))
        self.add_photo(folder / "photo.jpg", "photo.jpg")
        with patch.object(face_thumbnails, "delete_face_thumbnails", return_value=["face thumbnail locked"]):
            count, errors = remove_folder_and_index(str(folder))
        self.assertEqual((count, errors), (1, ["face thumbnail locked"]))
        self.assertEqual(database.get_photo_count(), 0)

    def test_removal_batches_more_than_500_photos(self):
        folder = self.root / "photos"
        database.add_folder(str(folder))
        with closing(database.get_connection()) as conn, conn:
            conn.executemany("INSERT INTO photos (path) VALUES (?)",
                             [(str(folder / f"{number}.jpg"),) for number in range(501)])
        self.assertEqual(remove_folder_and_index(str(folder)), (501, []))
        self.assertEqual(database.get_photo_count(), 0)

    def test_failed_transaction_leaves_folder_and_records(self):
        folder = self.root / "photos"
        database.add_folder(str(folder))
        path = folder / "photo.jpg"
        _, _, thumbnail = self.add_photo(path, "photo.jpg")
        with closing(database.get_connection()) as conn, conn:
            conn.execute("CREATE TRIGGER stop_delete BEFORE DELETE ON photos BEGIN SELECT RAISE(ABORT, 'no'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            remove_folder_and_index(str(folder))
        self.assertEqual(database.get_folders(), [str(folder)])
        self.assertEqual(database.get_photo_count(), 1)
        self.assertTrue(thumbnail.exists())
        with closing(database.get_connection()) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
