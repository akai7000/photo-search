import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from core import face_engine, face_models


class FaceEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_creates_cpu_detector_with_requested_threshold(self):
        with patch.object(face_engine, "runtime_models", return_value=(self.root / "yunet", self.root / "sface")), \
             patch.object(face_engine.cv2, "FaceDetectorYN_create") as detector, \
             patch.object(face_engine.cv2, "FaceRecognizerSF_create") as recognizer:
            face_engine._create_models()
        self.assertEqual(detector.call_args.args[3:6], (0.8, 0.3, 5000))
        self.assertEqual(detector.call_args.args[-1], face_engine.cv2.dnn.DNN_TARGET_CPU)
        self.assertEqual(recognizer.call_args.args[-1], face_engine.cv2.dnn.DNN_TARGET_CPU)

    def test_detect_resizes_landmarks_and_normalizes_sface_embedding(self):
        source = self.root / "photo.png"
        Image.new("RGB", (1280, 640), (10, 20, 30)).save(source)
        detection = np.array([[10, 5, 20, 10, 11, 6, 12, 7, 13, 8, 14, 9, 15, 10, 0.9]],
                             dtype=np.float32)
        detector, recognizer = MagicMock(), MagicMock()
        detector.detect.return_value = (1, detection)
        recognizer.feature.return_value = np.array([[3, 4]], dtype=np.float32)
        with patch.object(face_engine, "_models", (detector, recognizer)):
            results = face_engine.detect_faces(source)
        detector.setInputSize.assert_called_once_with((640, 320))
        aligned_image, aligned_face = recognizer.alignCrop.call_args.args
        np.testing.assert_array_equal(aligned_image[0, 0], [30, 20, 10])
        np.testing.assert_array_equal(aligned_face[0, :14], detection[0, :14] * 2)
        self.assertEqual(results[0]["bbox"], [20, 10, 60, 30])
        np.testing.assert_allclose(results[0]["embedding"], [0.6, 0.8])
        self.assertAlmostEqual(results[0]["det_score"], 0.9)

    def test_applies_exif_orientation(self):
        source = self.root / "oriented.png"
        image = Image.new("RGB", (2, 1))
        exif = image.getexif()
        exif[274] = 6
        image.save(source, exif=exif)
        detector = MagicMock()
        detector.detect.return_value = (1, None)
        with patch.object(face_engine, "_models", (detector, MagicMock())):
            self.assertEqual(face_engine.detect_faces(source), [])
        detector.setInputSize.assert_called_once_with((1, 2))

    def test_detection_errors_are_not_reported_as_no_faces(self):
        detector = MagicMock()
        with patch.object(face_engine, "_models", (detector, MagicMock())):
            with self.assertRaises(FileNotFoundError):
                face_engine.detect_faces(self.root / "missing.png")
        detector.detect.assert_not_called()

    def test_model_errors_are_reported(self):
        source = self.root / "photo.png"
        Image.new("RGB", (10, 10)).save(source)
        detector, recognizer = MagicMock(), MagicMock()
        detector.detect.return_value = (1, np.zeros((1, 15), dtype=np.float32))
        recognizer.feature.return_value = np.zeros((1, 2), dtype=np.float32)
        with patch.object(face_engine, "_models", (detector, recognizer)):
            with self.assertRaisesRegex(RuntimeError, "invalid embedding"):
                face_engine.detect_faces(source)

    def test_model_download_is_verified_and_invalid_download_is_not_installed(self):
        import io
        content = b"model"
        digest = hashlib.sha256(content).hexdigest()
        specs = {"one.onnx": ("first", digest), "two.onnx": ("second", digest)}
        with patch.object(face_models, "MODEL_SPECS", specs), \
             patch.object(face_models, "urlopen", side_effect=lambda *a, **k: io.BytesIO(content)):
            paths = face_models.ensure_models(self.root / "models", download=True)
            self.assertEqual([path.read_bytes() for path in paths], [content, content])
        (self.root / "models" / "two.onnx").unlink()
        with patch.object(face_models, "MODEL_SPECS", specs), \
             patch.object(face_models, "urlopen", return_value=io.BytesIO(b"bad")):
            with self.assertRaisesRegex(ValueError, "unexpected SHA-256"):
                face_models.ensure_models(self.root / "models", download=True)
        self.assertFalse((self.root / "models" / "two.onnx").exists())

    def test_bundled_missing_or_corrupt_model_fails(self):
        specs = {"one.onnx": ("first", hashlib.sha256(b"model").hexdigest()),
                 "two.onnx": ("second", hashlib.sha256(b"model").hexdigest())}
        with patch.object(face_models, "MODEL_SPECS", specs):
            with self.assertRaises(FileNotFoundError):
                face_models.ensure_models(self.root, download=False)
            (self.root / "one.onnx").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "unexpected SHA-256"):
                face_models.ensure_models(self.root, download=False)


if __name__ == "__main__":
    unittest.main()
