import argparse
import contextlib
import importlib.util
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from tools.benchmark_faces import detect_checked, percentile, select_images, sha256_file
from tools.benchmark_opencv_faces import detect_faces as detect_opencv_faces
from tools.benchmark_opencv_faces import benchmark as benchmark_opencv
from tools.benchmark_opencv_faces import score_threshold_arg

HAS_AURAFACE_DEPS = all(importlib.util.find_spec(name) for name in ("insightface", "onnxruntime"))
if HAS_AURAFACE_DEPS:
    from tools.benchmark_auraface import benchmark as benchmark_auraface
    from tools.benchmark_auraface import detect_faces as detect_auraface_faces


class FaceBenchmarkTests(unittest.TestCase):
    def test_opencv_score_threshold_is_validated_and_passed_to_detector(self):
        self.assertEqual(score_threshold_arg("0.7"), 0.7)
        for value in ("-0.1", "1.1", "nan"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                score_threshold_arg(value)
        with patch("tools.benchmark_opencv_faces.fingerprint_images", return_value=(["hash"], "dataset")), \
             patch("tools.benchmark_opencv_faces.sha256_file", return_value="hash"), \
             patch("tools.benchmark_opencv_faces.cv2.FaceDetectorYN_create") as detector, \
             patch("tools.benchmark_opencv_faces.cv2.FaceRecognizerSF_create"), \
             patch("tools.benchmark_opencv_faces.measure_samples", return_value=([], {})):
            result = benchmark_opencv([Path("photo.jpg")], Path("detector.onnx"), Path("sface.onnx"),
                                      1, 0, 640, 0.7)
        self.assertEqual(detector.call_args.args[3], 0.7)
        self.assertEqual(result["parameters"]["score_threshold"], 0.7)

    def test_selects_index_in_id_order_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = root / "first.jpg"
            second = root / "second.jpg"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            index = root / "index.db"
            with contextlib.closing(sqlite3.connect(index)) as connection:
                connection.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT)")
                connection.executemany("INSERT INTO photos VALUES (?, ?)", [(2, str(second)), (1, str(first))])
                connection.commit()
            before = sha256_file(index)
            self.assertEqual(select_images(None, index, 1), [first])
            self.assertEqual(sha256_file(index), before)

    def test_missing_indexed_image_fails_instead_of_skipping(self):
        with tempfile.TemporaryDirectory() as folder:
            index = Path(folder) / "index.db"
            with contextlib.closing(sqlite3.connect(index)) as connection:
                connection.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT)")
                connection.execute("INSERT INTO photos VALUES (1, ?)", (str(Path(folder) / "missing.jpg"),))
                connection.commit()
            with self.assertRaises(FileNotFoundError):
                select_images(None, index, 1)

    def test_detect_failure_is_not_counted_as_no_faces(self):
        def failed_detection(path):
            print(f"Face detection error for {path}: corrupt image")
            return []

        with self.assertRaisesRegex(RuntimeError, "corrupt image"):
            detect_checked(failed_detection, Path("bad.jpg"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(detect_checked(lambda path: [], Path("good.jpg")), [])

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([3, 1, 2], 0.95), 2.9)

    def test_opencv_rescales_detections_for_original_image_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.new("RGB", (200, 100)).save(path)
            detection = np.array([[10, 5, 20, 10, 11, 6, 12, 7, 13, 8, 14, 9, 15, 10, 0.9]],
                                 dtype=np.float32)
            detector = MagicMock()
            detector.detect.return_value = (1, detection)
            recognizer = MagicMock()
            recognizer.feature.return_value = np.array([[3, 4]], dtype=np.float32)

            faces = detect_opencv_faces(path, detector, recognizer, 100)

            detector.setInputSize.assert_called_once_with((100, 50))
            self.assertEqual(recognizer.alignCrop.call_args.args[0].shape, (100, 200, 3))
            np.testing.assert_array_equal(
                recognizer.alignCrop.call_args.args[1][0, :14], detection[0, :14] * 2
            )
            self.assertEqual(faces[0]["bbox"], [20, 10, 60, 30])
            np.testing.assert_allclose(faces[0]["embedding"], [0.6, 0.8])

    def test_opencv_invalid_embedding_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.new("RGB", (10, 10)).save(path)
            detector = MagicMock()
            detector.detect.return_value = (1, np.zeros((1, 15), dtype=np.float32))
            recognizer = MagicMock()
            recognizer.feature.return_value = np.zeros((1, 2), dtype=np.float32)
            with self.assertRaisesRegex(RuntimeError, "invalid embedding"):
                detect_opencv_faces(path, detector, recognizer, 0)

    @unittest.skipUnless(HAS_AURAFACE_DEPS, "AuraFace benchmark dependencies are optional")
    def test_auraface_receives_original_image_and_scaled_yunet_landmarks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.new("RGB", (200, 100), (10, 20, 30)).save(path)
            detection = np.array([[10, 5, 20, 10, 11, 6, 12, 7, 13, 8, 14, 9, 15, 10, 0.9]],
                                 dtype=np.float32)
            detector = MagicMock()
            detector.detect.return_value = (1, detection)
            recognizer = MagicMock()
            recognizer.get.return_value = np.array([3, 4], dtype=np.float32)

            faces = detect_auraface_faces(path, detector, recognizer, 100)

            image, face = recognizer.get.call_args.args
            self.assertEqual(image.shape, (100, 200, 3))
            np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
            np.testing.assert_array_equal(face.kps, detection[0, 4:14].reshape(5, 2) * 2)
            self.assertEqual(faces[0]["bbox"], [20, 10, 60, 30])
            np.testing.assert_allclose(faces[0]["embedding"], [0.6, 0.8])

    @unittest.skipUnless(HAS_AURAFACE_DEPS, "AuraFace benchmark dependencies are optional")
    def test_auraface_without_detections_does_not_run_recognizer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.new("RGB", (10, 10)).save(path)
            detector = MagicMock()
            detector.detect.return_value = (1, None)
            recognizer = MagicMock()
            self.assertEqual(detect_auraface_faces(path, detector, recognizer, 0), [])
            recognizer.get.assert_not_called()

    @unittest.skipUnless(HAS_AURAFACE_DEPS, "AuraFace benchmark dependencies are optional")
    def test_auraface_rejects_unavailable_cuda_provider(self):
        with patch("tools.benchmark_auraface.fingerprint_images", return_value=(["hash"], "dataset")), \
             patch("tools.benchmark_auraface.sha256_file", return_value="hash"), \
             patch("tools.benchmark_auraface.onnxruntime.get_available_providers",
                   return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "provider is unavailable"):
                benchmark_auraface([Path("photo.jpg")], Path("detector.onnx"), Path("model.onnx"),
                                   1, 0, 0, "CUDAExecutionProvider")


if __name__ == "__main__":
    unittest.main()
