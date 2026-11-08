"""Report application startup progress to the optional splash process."""

import atexit
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

_process = None
_failed = False


class WindowReadyGate:
    """Keep the splash until the native window is shown and its page is connected."""

    def __init__(self):
        self._events = set()
        self._lock = threading.Lock()

    def window_shown(self):
        self._mark("shown")

    def window_loaded(self):
        self._mark("loaded")

    def page_connected(self):
        self._mark("connected")

    def _mark(self, event):
        with self._lock:
            self._events.add(event)
            if {"shown", "loaded", "connected"} <= self._events and "closed" not in self._events:
                self._events.add("closed")
                close_splash()


def start_splash():
    global _process
    try:
        _process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name("startup_splash.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except OSError as exc:
        print(f"Could not open startup splash: {exc}", file=sys.stderr)
        return
    atexit.register(close_splash)
    ready = queue.Queue()
    output = _process.stdout
    threading.Thread(target=lambda: ready.put(output.readline()), daemon=True).start()
    try:
        message = ready.get(timeout=5)
        if message.strip() != "ready":
            print("Startup splash could not be displayed.", file=sys.stderr)
    except queue.Empty:
        print("Startup splash did not become ready in time.", file=sys.stderr)


def _send(kind, text=""):
    global _process
    if _process is None or _process.stdin is None:
        return
    try:
        _process.stdin.write(json.dumps({"kind": kind, "text": text}) + "\n")
        _process.stdin.flush()
    except (BrokenPipeError, OSError, ValueError) as exc:
        print(f"Startup splash is unavailable: {exc}", file=sys.stderr)
        _process = None


def report(message):
    _send("status", message)


def fail(exc_type, exc, traceback):
    global _failed
    _failed = True
    _send("error", f"{exc_type.__name__}: {exc}")
    sys.__excepthook__(exc_type, exc, traceback)


def close_splash():
    global _process
    if _process is None:
        return
    if not _failed:
        _send("close")
    if _process is not None and _process.stdin is not None:
        try:
            _process.stdin.close()
        except OSError as exc:
            print(f"Could not close startup splash connection: {exc}", file=sys.stderr)
    _process = None
