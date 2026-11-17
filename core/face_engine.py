"""YuNet face detection and SFace recognition used while indexing photos."""

import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from core.face_models import runtime_models

_lock = threading.Lock()
_models = None


def _create_models():
    detector_model, recognizer_model = runtime_models()
    cv2.ocl.setUseOpenCL(False)
    detector = cv2.FaceDetectorYN_create(
        str(detector_model), "", (320, 320), 0.5, 0.3, 5000,
        cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    recognizer = cv2.FaceRecognizerSF_create(
        str(recognizer_model), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    if detector is None or recognizer is None:
        raise RuntimeError("OpenCV could not load YuNet and SFace")
    return detector, recognizer


def detect_yunet_faces(path, detector, recognize, max_dimension, recognizer_name):
    with Image.open(path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))[:, :, ::-1].copy()

    height, width = image.shape[:2]
    if max_dimension and max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        input_width = max(1, round(width * scale))
        input_height = max(1, round(height * scale))
        detection_image = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)
    else:
        input_width, input_height = width, height
        detection_image = image

    detector.setInputSize((input_width, input_height))
    _, detections = detector.detect(detection_image)
    if detections is None:
        return []

    results = []
    for detection in detections:
        detection = detection.copy()
        detection[[0, 2, 4, 6, 8, 10, 12]] *= width / input_width
        detection[[1, 3, 5, 7, 9, 11, 13]] *= height / input_height
        embedding = np.asarray(recognize(image, detection), dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm == 0:
            raise RuntimeError(f"{recognizer_name} produced an invalid embedding for {path}")
        x, y, box_width, box_height = detection[:4].astype(int).tolist()
        results.append({
            "bbox": [x, y, x + box_width, y + box_height],
            "embedding": embedding / norm,
            "det_score": float(detection[14]),
        })
    return results


def detect_faces(image_path: str | Path) -> list[dict]:
    """Detect faces and return normalized embeddings and top-left/bottom-right boxes."""
    global _models
    with _lock:
        if _models is None:
            _models = _create_models()
        detector, recognizer = _models

        def recognize(image, detection):
            aligned = recognizer.alignCrop(image, detection.reshape(1, -1))
            return recognizer.feature(aligned)

        return detect_yunet_faces(image_path, detector, recognize, 640, "SFace")
