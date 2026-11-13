"""Small independent window shown while the application starts."""

import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path

logo_item = None
try:
    from .version import APP_VERSION
except ImportError:
    from version import APP_VERSION

WIDTH, HEIGHT = 600, 340
PROGRESS_START, PROGRESS_END, PROGRESS_Y = 258, 558, 181
LOGO_PATH = Path(__file__).resolve().parent.parent / "images" / "snowfoxninjalogo.png"


def load_logo(root):
    from PIL import Image, ImageTk

    with Image.open(LOGO_PATH) as image:
        image.thumbnail((210, 210), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image, master=root)


def main():
    root = tk.Tk()
    root.overrideredirect(True)
    root.resizable(False, False)
    root.configure(bg="#081321")
    x = (root.winfo_screenwidth() - WIDTH) // 2
    y = (root.winfo_screenheight() - HEIGHT) // 2
    root.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    top, bottom = (7, 16, 30), (19, 42, 59)
    for y in range(HEIGHT):
        ratio = y / (HEIGHT - 1)
        color = "#" + "".join(f"{round(a + (b - a) * ratio):02x}" for a, b in zip(top, bottom))
        canvas.create_line(0, y, WIDTH, y, fill=color)

    canvas.create_rectangle(1, 1, WIDTH - 2, HEIGHT - 2, outline="#355a71")
    try:
        logo = load_logo(root)
    except (ImportError, OSError, ValueError, tk.TclError) as exc:
        print(f"Could not load splash image {LOGO_PATH}: {exc}", file=sys.stderr)
    else:
        logo_item = canvas.create_image(24, 58, image=logo, anchor="nw")
    canvas.create_text(258, 65, text="PHOTO SEARCH", fill="#91bbd0",
                       font=("Segoe UI", 9, "bold"), anchor="nw")
    canvas.create_text(258, 88, text="Your photos.\nEasier to find.", fill="#f3f8fc",
                       font=("Segoe UI", 20, "bold"), anchor="nw")
    canvas.create_text(558, 34, text=f"v{APP_VERSION}", fill="#abc3d2",
                       font=("Segoe UI", 10), anchor="ne")
    canvas.create_line(PROGRESS_START, PROGRESS_Y, PROGRESS_END, PROGRESS_Y,
                       fill="#334e63", width=6, capstyle="round")
    progress = canvas.create_line(PROGRESS_START, PROGRESS_Y, PROGRESS_START, PROGRESS_Y,
                                  fill="#7eb8d4", width=6, capstyle="round")
    heading = canvas.create_text(258, 201, text="GETTING THINGS READY", fill="#e9b55f",
                                 font=("Segoe UI", 9, "bold"), anchor="nw")
    status = canvas.create_text(258, 226, text="Preparing application...", fill="#dbe8f0",
                                font=("Segoe UI", 10), anchor="nw", width=300)
    dismiss = tk.Button(root, text="Close", command=root.destroy, bg="#a9d2e5", fg="#0a1a2a",
                        activebackground="#e9f4f9", activeforeground="#0a1a2a",
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
        start = max(PROGRESS_START, PROGRESS_START + position)
        end = min(PROGRESS_END, PROGRESS_START + position + 130)
        canvas.coords(progress, start, PROGRESS_Y, max(start, end), PROGRESS_Y)
        position += 5
        if position > PROGRESS_END - PROGRESS_START:
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
                canvas.coords(progress, PROGRESS_START, PROGRESS_Y, PROGRESS_END, PROGRESS_Y)
                canvas.itemconfigure(progress, fill="#eaa09c")
                if logo_item is not None:
                    canvas.itemconfigure(logo_item, state="hidden")
                canvas.coords(heading, 40, 201)
                canvas.coords(status, 40, 226)
                canvas.itemconfigure(status, width=520)
                canvas.itemconfigure(heading, text="STARTUP FAILED", fill="#ffb5af")
                text = message.get("text") or "The application stopped before startup completed."
                canvas.itemconfigure(status, text=f"Could not start Photo Search.\n{text}", fill="#ffd1cc")
                dismiss.place(x=558, y=325, anchor="se")
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
