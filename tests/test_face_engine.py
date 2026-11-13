import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from core import face_engine


class FaceEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_detect_faces_returns_insightface_results_and_normalizes_embeddings(self):
        source = Path(self.temp_dir.name) / "photo.png"
        Image.new("RGB", (2, 1), (10, 20, 30)).save(source)
        detected_face = SimpleNamespace(
            bbox=np.array([1.2, 2.8, 10.1, 20.9]),
            embedding=np.array([3.0, 4.0], dtype=np.float32),
            det_score=0.95,
        )
        app = MagicMock()
        app.get.return_value = [detected_face]

        with patch.object(face_engine, "_get_app", return_value=app):
            results = face_engine.detect_faces(source)

        np.testing.assert_array_equal(app.get.call_args.args[0][0, 0], [30, 20, 10])
        self.assertEqual(results[0]["bbox"], [1, 2, 10, 20])
        np.testing.assert_allclose(results[0]["embedding"], [0.6, 0.8])
        self.assertEqual(results[0]["det_score"], 0.95)

    def test_detect_faces_skips_zero_embeddings(self):
        source = Path(self.temp_dir.name) / "photo.png"
        Image.new("RGB", (2, 1)).save(source)
        app = MagicMock()
        app.get.return_value = [SimpleNamespace(
            bbox=np.array([1, 2, 10, 20]),
            embedding=np.zeros(2, dtype=np.float32),
            det_score=0.95,
        )]

        with patch.object(face_engine, "_get_app", return_value=app):
            self.assertEqual(face_engine.detect_faces(source), [])

    def test_detect_faces_applies_exif_orientation(self):
        source = Path(self.temp_dir.name) / "oriented.png"
        image = Image.new("RGB", (2, 1))
        exif = image.getexif()
        exif[274] = 6
        image.save(source, exif=exif)
        app = MagicMock()
        app.get.return_value = []

        with patch.object(face_engine, "_get_app", return_value=app):
            self.assertEqual(face_engine.detect_faces(source), [])

        self.assertEqual(app.get.call_args.args[0].shape, (2, 1, 3))

    def test_missing_image_returns_no_faces_without_loading_the_model(self):
        app_loader = MagicMock()
        with patch.object(face_engine, "_get_app", app_loader):
            self.assertEqual(face_engine.detect_faces(Path(self.temp_dir.name) / "missing.png"), [])
        app_loader.assert_not_called()
