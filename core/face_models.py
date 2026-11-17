"""Acquire and verify the YuNet and SFace models used by the face pipeline."""

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from core.data_location import PROJECT_ROOT

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODEL_SPECS = {
    "face_detection_yunet_2026may.onnx": (
        "face_detection_yunet",
        "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def verify_model(path: Path, expected_hash: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"Face model has an unexpected SHA-256: {path}")


def ensure_models(directory: Path, *, download: bool) -> tuple[Path, Path]:
    if download:
        directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (model_dir, expected_hash) in MODEL_SPECS.items():
        path = directory / name
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Bundled face model is missing: {path}")
            url = (
                f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}"
                f"/models/{model_dir}/{name}"
            )
            temp_path = None
            try:
                with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
                    dir=directory, delete=False
                ) as temp:
                    temp_path = Path(temp.name)
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        temp.write(chunk)
                verify_model(temp_path, expected_hash)
                os.replace(temp_path, path)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        verify_model(path, expected_hash)
        paths.append(path)
    return paths[0], paths[1]


def runtime_models() -> tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        return ensure_models(Path(sys._MEIPASS) / "models", download=False)
    return ensure_models(PROJECT_ROOT / "models", download=True)
