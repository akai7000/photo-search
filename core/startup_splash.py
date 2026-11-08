"""Small independent window shown while the application starts."""

import json
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk


def main():
    root = tk.Tk()
    root.title("Photo Search")
    root.resizable(False, False)
    root.configure(bg="#1c1c1c")
    width, height = 560, 300
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    frame = tk.Frame(root, bg="#1c1c1c", padx=28, pady=24)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, text="Photo Search", fg="white", bg="#1c1c1c",
             font=("Segoe UI", 20, "bold"), anchor="w").pack(fill="x")
    tk.Label(frame, text="Starting up...", fg="#cbd5e1", bg="#1c1c1c",
             font=("Segoe UI", 11), anchor="w").pack(fill="x", pady=(8, 12))
    progress = ttk.Progressbar(frame, mode="indeterminate")
    progress.pack(fill="x")
    progress.start(12)
    status = tk.Label(frame, text="Preparing application...", fg="#e2e8f0", bg="#1c1c1c",
                      font=("Segoe UI", 10), anchor="nw", justify="left", wraplength=500)
    status.pack(fill="both", expand=True, pady=(18, 0))
    messages = queue.Queue()

    def read_messages():
        for line in sys.stdin:
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError as exc:
                messages.put({"kind": "error", "text": f"Invalid startup message: {exc}"})
        messages.put({"kind": "eof"})

    threading.Thread(target=read_messages, daemon=True).start()
    root.update()
    print("ready", flush=True)
    highlights = []
    failed = False

    def update():
        nonlocal failed
        while True:
            try:
                message = messages.get_nowait()
            except queue.Empty:
                break
            kind = message.get("kind")
            if kind == "close":
                root.destroy()
                return
            if kind == "error" or (kind == "eof" and not failed):
                failed = True
                progress.stop()
                text = message.get("text") or "The application stopped before startup completed."
                status.config(text=f"Could not start Photo Search.\n{text}\n\nClose this window to dismiss.", fg="#ff9999")
            elif kind == "status" and not failed:
                text = message["text"]
                if text == "Database initialized." or text.startswith(("Using GPU:", "Using CPU")):
                    highlights.append(text)
                lines = highlights + ([] if text in highlights else [text])
                status.config(text="\n".join(lines))
        root.after(100, update)

    root.after(50, update)
    root.mainloop()


if __name__ == "__main__":
    main()
