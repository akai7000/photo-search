"""InsightFace face detection and recognition used while indexing photos."""

import threading
from importlib import import_module
from pathlib import Path

from PIL import Image, ImageOps

_app_lock = threading.Lock()
_app = None


def _create_app():
    import_module("torch")
    import insightface

    app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _get_app():
    global _app
    with _app_lock:
        if _app is None:
            _app = _create_app()
        return _app


def detect_faces(image_path: str | Path) -> list[dict]:
    """Detect faces and return normalized embeddings and top-left/bottom-right boxes."""
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        app = _get_app()
        import numpy as np

        image = np.asarray(image)[:, :, ::-1].copy()
        results = []
        for face in app.get(image):
            embedding = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if not np.isfinite(norm) or norm == 0:
                continue
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": embedding / norm,
                "det_score": float(face.det_score),
            })
        return results
    except Exception as exc:
        print(f"Face detection error for {image_path}: {exc}")
        return []
