"""Initialize streams and child processes in a frozen Windows application."""

import multiprocessing
import os
import sys
from pathlib import Path


def _pipe_stream(handle_id, mode):
    import ctypes
    import msvcrt

    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    handle = kernel32.GetStdHandle(handle_id)
    if not handle or handle == ctypes.c_void_p(-1).value:
        raise OSError(f"Missing inherited standard handle {handle_id}")
    fd = msvcrt.open_osfhandle(handle, os.O_BINARY)
    return os.fdopen(fd, mode, encoding="utf-8", buffering=1)


def initialize():
    multiprocessing.freeze_support()
    if sys.argv[1:] == ["--photo-search-splash"]:
        sys.stdin = _pipe_stream(-10, "r")
        sys.stdout = _pipe_stream(-11, "w")
        from core import startup_splash
        startup_splash.main()
        raise SystemExit(0)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is required for installed Photo Search")
    log_dir = Path(local_app_data) / "Photo Search"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NICEGUI_STORAGE_PATH", str(log_dir / "nicegui"))
    log = (log_dir / "photo-search.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log
