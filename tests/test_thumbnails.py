import importlib.util
import os
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from PIL import Image

from core import database


def load_scanner():
    clip = ModuleType("core.clip_engine")
    clip.get_image_embedding = MagicMock(return_value=None)
    faces = ModuleType("core.face_engine")
    faces.detect_faces = MagicMock(return_value=[])
    path = Path(__file__).resolve().parents[1] / "core" / "scanner.py"
    spec = importlib.util.spec_from_file_location("thumbnail_test_scanner", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"core.clip_engine": clip, "core.face_engine": faces}):
        spec.loader.exec_module(module)
    return module


class ThumbnailTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.scanner = load_scanner()
        self.db_patch = patch.object(database, "DB_PATH", self.root / "test.db")
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        database.init_db()
        self.thumb_dir = self.root / "thumbnails"
        self.thumb_patch = patch.object(self.scanner, "THUMBNAIL_DIR", self.thumb_dir, create=True)
        self.thumb_patch.start()
        self.addCleanup(self.thumb_patch.stop)

    def make_photo(self, path, orientation=None):
        image = Image.new("RGB", (80, 40))
        for box, color in [
            ((0, 0, 40, 20), (255, 0, 0)),
            ((40, 0, 80, 20), (0, 255, 0)),
            ((0, 20, 40, 40), (0, 0, 255)),
            ((40, 20, 80, 40), (255, 255, 0)),
        ]:
            image.paste(color, box)
        exif = image.getexif()
        if orientation is not None:
            exif[274] = orientation
        image.save(path, exif=exif)

    def assert_corners(self, path, expected):
        colors = {"R": (255, 0, 0), "G": (0, 255, 0),
                  "B": (0, 0, 255), "Y": (255, 255, 0)}
        with Image.open(path) as image:
            width, height = image.size
            actual = [
                image.getpixel((width // 4, height // 4)),
                image.getpixel((3 * width // 4, height // 4)),
                image.getpixel((width // 4, 3 * height // 4)),
                image.getpixel((3 * width // 4, 3 * height // 4)),
            ]
            for pixel, name in zip(actual, expected):
                self.assertTrue(all(abs(a - b) < 30 for a, b in zip(pixel, colors[name])),
                                (pixel, name))
            self.assertNotIn(274, image.getexif())

    def test_all_exif_orientations(self):
        orientations = {
            None: "RGBY", 1: "RGBY", 2: "GRYB", 3: "YBGR", 4: "BYRG",
            5: "RBGY", 6: "BRYG", 7: "YGBR", 8: "GYRB",
        }
        for orientation, corners in orientations.items():
            with self.subTest(orientation=orientation):
                source = self.photos / f"photo_{orientation}.png"
                self.make_photo(source, orientation)
                thumbnail, width, height = self.scanner.generate_thumbnail(source, self.thumb_dir)
                self.assertEqual((width, height), (80, 40))
                self.assert_corners(thumbnail, corners)
                with Image.open(thumbnail) as result:
                    self.assertEqual(result.size, (40, 80) if orientation in (5, 6, 7, 8) else (80, 40))

    def test_jpeg_orientation_is_applied_before_resizing(self):
        source = self.photos / "photo.jpg"
        self.make_photo(source, 6)
        thumbnail, width, height = self.scanner.generate_thumbnail(source, self.thumb_dir, (30, 30))
        self.assertEqual((width, height), (80, 40))
        self.assert_corners(thumbnail, "BRYG")
        with Image.open(thumbnail) as result:
            self.assertEqual(result.size, (15, 30))
        with Image.open(source) as original:
            self.assertEqual(original.getexif()[274], 6)

    def test_rescan_refreshes_existing_thumbnails_without_reindexing(self):
        source = self.photos / "photo.png"
        self.make_photo(source, 6)
        old_thumbnail = self.root / "old.jpg"
        Image.new("RGB", (80, 40), "red").save(old_thumbnail)
        database.insert_or_update_photo({
            "path": str(source), "file_hash": "existing", "file_size": source.stat().st_size,
            "modified_time": source.stat().st_mtime, "width": 80, "height": 40,
            "thumbnail_path": str(old_thumbnail), "clip_embedding": b"existing",
        })

        result = self.scanner.scan_folders([str(self.photos)])

        self.assertEqual((result["added"], result["updated"], result["skipped"], result["errors"]), (0, 0, 1, 0))
        with closing(database.get_connection()) as conn:
            row = conn.execute("SELECT * FROM photos WHERE path = ?", (str(source),)).fetchone()
        self.assertNotEqual(row["thumbnail_path"], str(old_thumbnail))
        self.assertEqual(row["clip_embedding"], b"existing")
        self.assertEqual((row["width"], row["height"]), (80, 40))
        self.assert_corners(row["thumbnail_path"], "BRYG")
        self.scanner.get_image_embedding.assert_not_called()
        self.scanner.detect_faces.assert_not_called()
        with patch.object(self.scanner, "generate_thumbnail", wraps=self.scanner.generate_thumbnail) as generate:
            self.scanner.scan_folders([str(self.photos)])
            generate.assert_not_called()
        Path(row["thumbnail_path"]).unlink()
        result = self.scanner.scan_folders([str(self.photos)])
        self.assertEqual(result["errors"], 0)
        self.assert_corners(row["thumbnail_path"], "BRYG")

    def test_rescan_reuses_thumbnail_already_in_selected_data_directory(self):
        source = self.photos / "photo.png"
        self.make_photo(source)
        expected, _, _ = self.scanner.generate_thumbnail(source, self.thumb_dir)
        database.insert_or_update_photo({
            "path": str(source), "modified_time": source.stat().st_mtime,
            "thumbnail_path": str(self.root / "previous" / "thumbnails" / Path(expected).name),
        })
        with patch.object(self.scanner, "generate_thumbnail") as generate:
            result = self.scanner.scan_folders([str(self.photos)])
            generate.assert_not_called()
        self.assertEqual(result["skipped"], 1)
        with closing(database.get_connection()) as conn:
            self.assertEqual(conn.execute(
                "SELECT thumbnail_path FROM photos WHERE path = ?", (str(source),)
            ).fetchone()[0], expected)

    def test_database_migration_forces_non_insightface_faces_to_be_reindexed(self):
        source = self.photos / "photo.png"
        self.make_photo(source)
        database.add_folder(str(self.photos))
        photo_id = database.insert_or_update_photo({
            "path": str(source),
            "modified_time": source.stat().st_mtime,
            "thumbnail_path": str(self.root / "old.jpg"),
        })
        database.insert_face(photo_id, [1, 2, 20, 30], b"old embedding")
        database.set_folder_total(str(self.photos), 1)
        database.mark_folder_indexed(str(self.photos))
        with closing(database.get_connection()) as conn:
            conn.execute(
                "ALTER TABLE photos ADD COLUMN face_backend TEXT NOT NULL DEFAULT 'insightface'"
            )
            conn.execute(
                "UPDATE photos SET face_backend = 'alternate' WHERE id = ?", (photo_id,)
            )
            conn.commit()

        database.init_db()

        with patch.object(self.scanner, "detect_faces", return_value=[]) as detect:
            result = self.scanner.scan_folders(
                [str(self.photos)],
                skip_indexed_folders=True,
            )

        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["folders_skipped"], 0)
        detect.assert_called_once_with(source)
        with closing(database.get_connection()) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(photos)")}
            faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            row = conn.execute(
                "SELECT modified_time FROM photos WHERE id = ?", (photo_id,)
            ).fetchone()
        self.assertNotIn("face_backend", columns)
        self.assertEqual(faces, 0)
        self.assertEqual(row["modified_time"], source.stat().st_mtime)

    def test_cancel_stops_scan_and_leaves_folder_unmarked(self):
        import threading
        self.make_photo(self.photos / "a.png", 1)
        self.make_photo(self.photos / "b.png", 1)
        cancel = threading.Event()
        cancel.set()

        result = self.scanner.scan_folders([str(self.photos)], cancel_event=cancel)

        self.assertTrue(result["cancelled"])
        self.assertEqual(result["added"] + result["skipped"] + result["errors"], 0)
        self.assertEqual(database.get_folder_status(str(self.photos)), "Not indexed")

    def test_folder_index_date_changes_only_for_folders_with_changes(self):
        other = self.root / "other"
        empty = self.root / "empty"
        other.mkdir()
        empty.mkdir()
        folders = [str(self.photos), str(other), str(empty)]
        for folder in folders:
            database.add_folder(folder)
        self.make_photo(self.photos / "first.png")
        self.make_photo(other / "second.png")
        self.scanner.scan_folders(folders)
        self.assertIsNotNone(database.get_folder_progress(folders[0])["indexed_at"])
        self.assertIsNone(database.get_folder_progress(folders[2])["indexed_at"])
        self.assertEqual(database.get_folder_status(folders[2]), "Indexed")
        with closing(database.get_connection()) as conn:
            conn.execute("UPDATE folders SET indexed_at = '2000-01-01T00:00:00' WHERE path IN (?, ?)", folders[:2])
            conn.commit()
        (self.thumb_dir / self.scanner.thumbnail_path_for(self.photos / "first.png", self.thumb_dir).name).unlink()
        self.make_photo(other / "new.png")
        result = self.scanner.scan_folders(folders)
        self.assertEqual((result["added"], result["skipped"]), (1, 2))
        self.assertEqual(database.get_folder_progress(folders[0])["indexed_at"], "2000-01-01T00:00:00")
        self.assertNotEqual(database.get_folder_progress(folders[1])["indexed_at"], "2000-01-01T00:00:00")
        self.assertIsNone(database.get_folder_progress(folders[2])["indexed_at"])

    def test_all_skipped_scan_keeps_database_timestamps_and_updates_advance_them(self):
        source = self.photos / "first.png"
        database.add_folder(str(self.photos))
        self.make_photo(source)
        first = self.scanner.scan_folders([str(self.photos)])
        self.assertEqual((first["added"], first["updated"]), (1, 0))
        with closing(database.get_connection()) as conn:
            conn.execute("UPDATE folders SET indexed_at = ? WHERE path = ?",
                         ("2000-01-01T00:00:00", str(self.photos)))
            conn.execute("UPDATE photos SET indexed_at = ? WHERE path = ?",
                         ("2000-01-02T00:00:00", str(source)))
            conn.commit()
        skipped = self.scanner.scan_folders([str(self.photos)])
        self.assertEqual((skipped["added"], skipped["updated"], skipped["skipped"]), (0, 0, 1))
        with closing(database.get_connection()) as conn:
            folder_time = conn.execute("SELECT indexed_at FROM folders WHERE path = ?", (str(self.photos),)).fetchone()[0]
            photo_time = conn.execute("SELECT indexed_at FROM photos WHERE path = ?", (str(source),)).fetchone()[0]
        self.assertEqual(folder_time, "2000-01-01T00:00:00")
        self.assertEqual(photo_time, "2000-01-02T00:00:00")
        stat = source.stat()
        os.utime(source, (stat.st_atime, stat.st_mtime + 5))
        updated = self.scanner.scan_folders([str(self.photos)])
        self.assertEqual((updated["added"], updated["updated"]), (0, 1))
        with closing(database.get_connection()) as conn:
            folder_time = conn.execute("SELECT indexed_at FROM folders WHERE path = ?", (str(self.photos),)).fetchone()[0]
            photo_time = conn.execute("SELECT indexed_at FROM photos WHERE path = ?", (str(source),)).fetchone()[0]
        self.assertNotEqual(folder_time, "2000-01-01T00:00:00")
        self.assertNotEqual(photo_time, "2000-01-02T00:00:00")

    def test_cancel_records_changes_without_marking_folder_complete(self):
        import threading
        database.add_folder(str(self.photos))
        self.make_photo(self.photos / "a.png")
        self.make_photo(self.photos / "b.png")
        cancel = threading.Event()
        def on_progress(done, total, path, root):
            if done == 1:
                cancel.set()
        result = self.scanner.scan_folders([str(self.photos)], on_progress, cancel)
        self.assertTrue(result["cancelled"])
        self.assertIsNotNone(database.get_folder_progress(str(self.photos))["indexed_at"])
        self.assertEqual(database.get_folder_status(str(self.photos)), "Partially indexed (1 of 2)")

    def test_failed_scan_does_not_change_timestamp(self):
        database.add_folder(str(self.photos))
        self.make_photo(self.photos / "bad.png")
        with patch.object(self.scanner, "get_image_embedding", side_effect=RuntimeError("model error")):
            result = self.scanner.scan_folders([str(self.photos)])
        self.assertEqual(result["errors"], 1)
        self.assertIsNone(database.get_folder_progress(str(self.photos))["indexed_at"])

    def test_skip_completed_folder_without_traversal_then_rescan_changes(self):
        folder = str(self.photos)
        database.add_folder(folder)
        source = self.photos / "first.png"
        self.make_photo(source)
        self.scanner.scan_folders([folder])
        before = database.get_folder_progress(folder)
        stat = source.stat()
        os.utime(source, (stat.st_atime, stat.st_mtime + 5))
        self.make_photo(self.photos / "new.png")
        with patch.object(self.scanner.os, "walk", side_effect=AssertionError("must not traverse")):
            result = self.scanner.scan_folders([folder], skip_indexed_folders=True)
        self.assertEqual((result["folders_skipped"], result["added"], result["updated"]), (1, 0, 0))
        self.assertEqual(database.get_folder_progress(folder), before)
        result = self.scanner.scan_folders([folder], skip_indexed_folders=False)
        self.assertEqual((result["folders_skipped"], result["added"], result["updated"]), (0, 1, 1))

    def test_skip_empty_and_completed_folders_but_scan_new_and_partial(self):
        empty = self.root / "empty"
        new = self.root / "new"
        partial = self.root / "partial"
        for folder in (empty, new, partial, self.photos):
            folder.mkdir(exist_ok=True)
            database.add_folder(str(folder))
        self.make_photo(self.photos / "a.png")
        self.make_photo(new / "b.png")
        self.make_photo(partial / "c.png")
        self.make_photo(partial / "d.png")
        self.scanner.scan_folders([str(empty), str(self.photos)])
        database.set_folder_total(str(partial), 2)
        database.insert_or_update_photo({"path": str(partial / "c.png"), "modified_time": 0})
        result = self.scanner.scan_folders(
            list(map(str, (empty, self.photos, new, partial))), skip_indexed_folders=True)
        self.assertEqual((result["folders_skipped"], result["added"], result["updated"]), (2, 2, 1))

    def test_cancelled_and_failed_rescans_are_retried(self):
        import threading
        folder = str(self.photos)
        database.add_folder(folder)
        self.make_photo(self.photos / "a.png")
        self.scanner.scan_folders([folder])
        cancel = threading.Event()
        cancel.set()
        self.scanner.scan_folders([folder], cancel_event=cancel)
        self.assertFalse(database.get_folder_progress(folder)["scan_completed"])
        result = self.scanner.scan_folders([folder], skip_indexed_folders=True)
        self.assertEqual(result["folders_skipped"], 0)
        self.make_photo(self.photos / "bad.png")
        with patch.object(self.scanner, "get_image_embedding", side_effect=RuntimeError("model error")):
            result = self.scanner.scan_folders([folder])
        self.assertEqual(result["errors"], 1)
        self.assertFalse(database.get_folder_progress(folder)["scan_completed"])
        result = self.scanner.scan_folders([folder], skip_indexed_folders=True)
        self.assertEqual((result["folders_skipped"], result["added"]), (0, 1))

    def test_skip_selected_child_while_scanning_parent(self):
        child = self.photos / "child"
        child.mkdir()
        database.add_folder(str(self.photos))
        database.add_folder(str(child))
        self.make_photo(child / "old.png")
        self.scanner.scan_folders([str(child)])
        child_before = database.get_folder_progress(str(child))
        self.make_photo(child / "new.png")
        self.make_photo(self.photos / "parent.png")
        result = self.scanner.scan_folders([str(self.photos), str(child)], skip_indexed_folders=True)
        self.assertEqual((result["folders_skipped"], result["added"]), (1, 1))
        self.assertEqual(database.get_folder_progress(str(child)), child_before)
        with closing(database.get_connection()) as conn:
            self.assertIsNone(conn.execute("SELECT id FROM photos WHERE path = ?", (str(child / "new.png"),)).fetchone())
        self.assertEqual(database.get_folder_progress(str(self.photos))["total"], 2)

    def test_reset_clears_completed_empty_folder(self):
        database.add_folder(str(self.photos))
        self.scanner.scan_folders([str(self.photos)])
        database.delete_all_photos()
        result = self.scanner.scan_folders([str(self.photos)], skip_indexed_folders=True)
        self.assertEqual(result["folders_skipped"], 0)

    def test_failed_refresh_leaves_old_thumbnail_and_retries(self):
        source = self.photos / "photo.png"
        self.make_photo(source, 6)
        database.insert_or_update_photo({
            "path": str(source), "modified_time": source.stat().st_mtime,
            "thumbnail_path": "old.jpg",
        })
        with patch.object(self.scanner, "generate_thumbnail", return_value=(None, None, None)):
            result = self.scanner.scan_folders([str(self.photos)])
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["skipped"], 0)
        with closing(database.get_connection()) as conn:
            row = conn.execute("SELECT thumbnail_path FROM photos WHERE path = ?", (str(source),)).fetchone()
        self.assertEqual(row["thumbnail_path"], "old.jpg")
        result = self.scanner.scan_folders([str(self.photos)])
        self.assertEqual(result["errors"], 0)
        with closing(database.get_connection()) as conn:
            row = conn.execute("SELECT thumbnail_path FROM photos WHERE path = ?", (str(source),)).fetchone()
        self.assert_corners(row["thumbnail_path"], "BRYG")

    def test_same_stem_images_have_separate_thumbnails(self):
        first = self.photos / "one"
        second = self.photos / "two"
        first.mkdir()
        second.mkdir()
        self.make_photo(first / "photo.png", 3)
        self.make_photo(second / "photo.png", 6)
        result = self.scanner.scan_folders([str(self.photos)])
        self.assertEqual(result["errors"], 0)
        with closing(database.get_connection()) as conn:
            rows = conn.execute("SELECT path, thumbnail_path FROM photos ORDER BY path").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["thumbnail_path"], rows[1]["thumbnail_path"])
        self.assert_corners(rows[0]["thumbnail_path"], "YBGR")
        self.assert_corners(rows[1]["thumbnail_path"], "BRYG")


if __name__ == "__main__":
    unittest.main()
