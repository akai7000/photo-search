"""Persistent location of the application's data, independent of the index."""

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def storage_paths():
    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise RuntimeError("LOCALAPPDATA is required for installed Photo Search")
        root = Path(local_app_data) / "Photo Search"
        return root / "data", root / "config.json"
    return PROJECT_ROOT / "data", PROJECT_ROOT / ".photo-search-config.json"


DEFAULT_DATA_DIR, CONFIG_PATH = storage_paths()


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
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=CONFIG_PATH.parent, delete=False
        ) as file:
            temp_path = Path(file.name)
            json.dump({"data_directory": str(path)}, file)
        os.replace(temp_path, CONFIG_PATH)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
