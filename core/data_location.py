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


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration: {CONFIG_PATH}")
    return config


def load_data_directory() -> Path:
    if not CONFIG_PATH.exists():
        return DEFAULT_DATA_DIR
    config = load_config()
    if "data_directory" not in config:
        if any(key in config for key in (
            "download_url",
            "mock_license_key",
            "license_key",
            "license_machine_id",
            "license_last_validated",
            "license_expires_at",
            "license_permanent",
        )):
            return DEFAULT_DATA_DIR
        raise ValueError(f"Invalid data directory configuration: {CONFIG_PATH}")
    if not isinstance(config["data_directory"], str):
        raise ValueError(f"Invalid data directory configuration: {CONFIG_PATH}")
    path = Path(config["data_directory"])
    if not path.is_absolute():
        raise ValueError(f"Data directory must be an absolute path: {path}")
    if not path.is_dir():
        raise ValueError(f"Configured data directory is unavailable: {path}")
    return path


def save_data_directory(path: Path) -> None:
    save_config_value("data_directory", str(path))


def save_config_value(key: str, value: object) -> None:
    save_config_values({key: value})


def save_config_values(values: dict) -> None:
    config = load_config()
    config.update(values)
    _write_config(config)


def _write_config(config: dict) -> None:
    temp_path = None
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=CONFIG_PATH.parent, delete=False
        ) as file:
            temp_path = Path(file.name)
            json.dump(config, file)
        os.replace(temp_path, CONFIG_PATH)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
