import importlib.util
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
