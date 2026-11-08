import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from core.photo_preview import create_photo_preview


class PhotoPreviewTests(unittest.TestCase):
    def test_preview_is_bounded_and_respects_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.jpg"
            original = Image.new("RGB", (2400, 1200), "blue")
            exif = original.getexif()
            exif[274] = 6
            original.save(path, exif=exif)
            data = create_photo_preview(path)
            self.assertTrue(data.startswith("data:image/jpeg;base64,"))
            with Image.open(BytesIO(base64.b64decode(data.split(",", 1)[1]))) as preview:
                self.assertEqual(preview.size, (800, 1600))
                self.assertIsNone(preview.getexif().get(274))

    def test_preview_does_not_enlarge_small_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.png"
            Image.new("RGB", (80, 40)).save(path)
            data = create_photo_preview(path)
            with Image.open(BytesIO(base64.b64decode(data.split(",", 1)[1]))) as preview:
                self.assertEqual(preview.size, (80, 40))

    def test_missing_source_raises(self):
        with self.assertRaises(FileNotFoundError):
            create_photo_preview(Path(tempfile.gettempdir()) / "missing-photo-preview-test.jpg")


if __name__ == "__main__":
    unittest.main()
