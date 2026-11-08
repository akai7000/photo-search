import torch  # noqa: F401  Must load before onnxruntime so it can reuse torch's CUDA/cuDNN DLLs.
import insightface
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

# Load model once
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

def detect_faces(image_path: str | Path) -> List[Dict]:
    """
    Detect faces in an image.
    Returns a list of dicts with bbox, embedding, and confidence.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        faces = app.get(img)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            embedding = face.embedding.astype(np.float32)
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            results.append({
                "bbox": bbox,
                "embedding": embedding,
                "det_score": float(face.det_score),
            })

        return results
    except Exception as e:
        print(f"Face detection error for {image_path}: {e}")
        return []