"""Small independent window shown while the application starts."""

import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path

try:
    from .version import APP_VERSION
except ImportError:
    from version import APP_VERSION

WIDTH, HEIGHT = 600, 340
PROGRESS_Y = 116
LOGO_PATH = Path(__file__).resolve().parent.parent / "images" / "photo_search.jpg"


def load_logo(root):
    from PIL import Image, ImageTk

    with Image.open(LOGO_PATH) as image:
        image.thumbnail((42, 42), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image, master=root)


def main():
    root = tk.Tk()
    root.overrideredirect(True)
    root.resizable(False, False)
    root.configure(bg="#101c2e")
    x = (root.winfo_screenwidth() - WIDTH) // 2
    y = (root.winfo_screenheight() - HEIGHT) // 2
    root.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    top, bottom = (16, 25, 45), (20, 52, 65)
    for y in range(HEIGHT):
        ratio = y / (HEIGHT - 1)
        color = "#" + "".join(f"{round(a + (b - a) * ratio):02x}" for a, b in zip(top, bottom))
        canvas.create_line(0, y, WIDTH, y, fill=color)

    canvas.create_rectangle(1, 1, WIDTH - 2, HEIGHT - 2, outline="#345361")
    try:
        logo = load_logo(root)
    except (ImportError, OSError, ValueError, tk.TclError) as exc:
        print(f"Could not load splash image {LOGO_PATH}: {exc}", file=sys.stderr)
    else:
        canvas.create_image(39, 39, image=logo, anchor="nw")
    canvas.create_text(101, 42, text="Photo Search", fill="#f3f8fa",
                       font=("Segoe UI", 22, "bold"), anchor="nw")
    canvas.create_text(103, 78, text="YOUR PHOTO LIBRARY, READY TO EXPLORE",
                       fill="#a5c4ca", font=("Segoe UI", 9), anchor="nw")
    canvas.create_text(560, 46, text=f"v{APP_VERSION}", fill="#a5c4ca",
                       font=("Segoe UI", 12), anchor="ne")
    canvas.create_line(45, PROGRESS_Y, 555, PROGRESS_Y, fill="#385460", width=8, capstyle="round")
    progress = canvas.create_line(45, PROGRESS_Y, 45, PROGRESS_Y, fill="#75ded5", width=8, capstyle="round")
    heading = canvas.create_text(40, 135, text="GETTING THINGS READY", fill="#78d3cd",
                                 font=("Segoe UI", 9, "bold"), anchor="nw")
    status = canvas.create_text(40, 160, text="Preparing application...", fill="#e1edf0",
                                font=("Segoe UI", 10), anchor="nw", width=520)
    dismiss = tk.Button(root, text="Close", command=root.destroy, bg="#e4f1f0", fg="#142d39",
                        activebackground="#ffffff", activeforeground="#142d39",
                        relief="flat", padx=18, pady=4, font=("Segoe UI", 10))
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
    position = -130

    def animate():
        nonlocal position
        if failed:
            return
        start = max(45, 45 + position)
        end = min(555, 45 + position + 130)
        canvas.coords(progress, start, PROGRESS_Y, max(start, end), PROGRESS_Y)
        position += 5
        if position > 510:
            position = -130
        root.after(25, animate)

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
                canvas.coords(progress, 45, PROGRESS_Y, 555, PROGRESS_Y)
                canvas.itemconfigure(progress, fill="#eaa09c")
                canvas.itemconfigure(heading, text="STARTUP FAILED", fill="#ffb5af")
                text = message.get("text") or "The application stopped before startup completed."
                canvas.itemconfigure(status, text=f"Could not start Photo Search.\n{text}", fill="#ffd1cc")
                dismiss.place(x=560, y=283, anchor="ne")
            elif kind == "status" and not failed:
                text = message["text"]
                if text == "Database initialized." or text.startswith(("Using GPU:", "Using CPU")):
                    highlights.append(text)
                lines = highlights + ([] if text in highlights else [text])
                canvas.itemconfigure(status, text="\n".join(lines))
        root.after(100, update)

    root.after(25, animate)
    root.after(50, update)
    root.mainloop()


if __name__ == "__main__":
    main()
