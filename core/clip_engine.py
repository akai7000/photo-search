import torch, logging, warnings
import os

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "0")) or torch.get_num_threads())
import open_clip
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import numpy as np
from core.startup import report as report_startup

# Suppress the unauthenticated HF Hub warning
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

# Load model once (global)
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    device_message = f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
else:
    device_message = "Using CPU (CUDA not available)"
print(device_message)
report_startup(device_message)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def get_image_embedding(image_path: str | Path) -> np.ndarray | None:
    """Return a normalized CLIP embedding for an image."""
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype(np.float32)[0]
    except Exception as e:
        print(f"CLIP error for {image_path}: {e}")
        return None

def get_text_embedding(text: str) -> np.ndarray:
    """Return a normalized CLIP embedding for text."""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype(np.float32)[0]

def search_photos(query: str, top_k: int = 50, min_score: float = 0.2) -> List[Tuple[int, float]]:
    """
    Search photos by text query.
    Returns a list of (photo_id, similarity_score) sorted by score descending.
    """
    from core.database import get_connection
    import sqlite3

    query_embedding = get_text_embedding(query)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, clip_embedding FROM photos WHERE clip_embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        photo_id = row["id"]
        emb_bytes = row["clip_embedding"]
        if not emb_bytes:
            continue
        photo_emb = np.frombuffer(emb_bytes, dtype=np.float32)
        # Cosine similarity (embeddings are already normalized)
        similarity = float(np.dot(query_embedding, photo_emb))
        results.append((photo_id, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    # Only keep results above the threshold
    filtered = [(pid, score) for pid, score in results if score >= min_score]
    return filtered[:top_k]