"""Persistent location of the application's data, independent of the index."""

import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / ".photo-search-config.json"


def load_data_directory() -> Path:
    if not CONFIG_PATH.exists():
        return DEFAULT_DATA_DIR
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict) or not isinstance(config.get("data_directory"), str):
        raise ValueError(f"Invalid data directory configuration: {CONFIG_PATH}")
    path = Path(config["data_directory"])
    if not path.is_absolute():
        raise ValueError(f"Data directory must be an absolute path: {path}")
    if not path.is_dir():
        raise ValueError(f"Configured data directory is unavailable: {path}")
    return path


def save_data_directory(path: Path) -> None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=CONFIG_PATH.parent, delete=False
        ) as file:
            temp_path = Path(file.name)
            json.dump({"data_directory": str(path)}, file)
        os.replace(temp_path, CONFIG_PATH)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
