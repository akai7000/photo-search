import json
import itertools
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core import startup
from core import startup_splash


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.process = MagicMock()
        self.process.stdin = MagicMock()
        self.process_patch = patch.object(startup, "_process", self.process)
        self.failed_patch = patch.object(startup, "_failed", False)
        self.process_patch.start()
        self.failed_patch.start()
        self.addCleanup(self.process_patch.stop)
        self.addCleanup(self.failed_patch.stop)

    def test_progress_and_close_send_ordered_messages(self):
        startup.report("Database initialized.")
        startup.report("Using GPU: Test GPU (CUDA 12.4)")
        startup.close_splash()
        messages = [json.loads(call.args[0]) for call in self.process.stdin.write.call_args_list]
        self.assertEqual(messages, [
            {"kind": "status", "text": "Database initialized."},
            {"kind": "status", "text": "Using GPU: Test GPU (CUDA 12.4)"},
            {"kind": "close", "text": ""},
        ])
        self.process.stdin.close.assert_called_once()

    def test_failure_is_reported_and_left_visible(self):
        with patch.object(startup.sys, "__excepthook__") as hook:
            startup.fail(ValueError, ValueError("model failed"), None)
            hook.assert_called_once()
        startup.close_splash()
        messages = [json.loads(call.args[0]) for call in self.process.stdin.write.call_args_list]
        self.assertEqual(messages, [{"kind": "error", "text": "ValueError: model failed"}])
        self.process.stdin.close.assert_called_once()

    def test_closed_splash_does_not_prevent_startup(self):
        self.process.stdin.write.side_effect = BrokenPipeError("closed")
        with patch.object(startup.sys, "stderr"):
            startup.report("Loading...")
        self.assertIsNone(startup._process)

    def test_splash_waits_for_window_visibility_load_and_page_connection_in_any_order(self):
        events = ("window_shown", "window_loaded", "page_connected")
        for order in itertools.permutations(events):
            with self.subTest(order=order), patch.object(startup, "close_splash") as close:
                gate = startup.WindowReadyGate()
                for event in order[:2]:
                    getattr(gate, event)()
                    close.assert_not_called()
                getattr(gate, order[2])()
                close.assert_called_once()
                for event in order:
                    getattr(gate, event)()
                close.assert_called_once()

    def test_splash_has_no_title_bar_and_error_can_be_dismissed(self):
        root = MagicMock()
        canvas = MagicMock()
        close_button = MagicMock()
        with patch.object(startup_splash.tk, "Tk", return_value=root), \
             patch.object(startup_splash.tk, "Canvas", return_value=canvas), \
             patch.object(startup_splash.tk, "Button", return_value=close_button) as button, \
             patch.object(startup_splash, "load_logo") as logo, \
             patch.object(startup_splash.sys, "stdin", io.StringIO('{"kind":"error","text":"failed"}\n')), \
             patch.object(startup_splash.threading, "Thread") as thread:
            thread.side_effect = lambda target, daemon: MagicMock(start=target)
            startup_splash.main()
            root.overrideredirect.assert_called_once_with(True)
            logo.assert_called_once_with(root)
            canvas.create_image.assert_called_once_with(39, 39, image=logo.return_value, anchor="nw")
            canvas.create_text.assert_any_call(
                560, 46, text=f"v{startup_splash.APP_VERSION}", fill="#a5c4ca",
                font=("Segoe UI", 10), anchor="ne"
            )
            canvas.create_line.assert_any_call(
                45, 116, 555, 116, fill="#385460", width=8, capstyle="round"
            )
            self.assertFalse(any(call.kwargs.get("text") == "STARTING PHOTO SEARCH"
                                 for call in canvas.create_text.call_args_list))
            self.assertEqual(button.call_args.args[0], root)
            self.assertEqual(button.call_args.kwargs["command"], root.destroy)
            update = root.after.call_args.args[1]
            update()
            close_button.place.assert_called_once()
            canvas.itemconfigure.assert_any_call(
                canvas.create_text.return_value, text="STARTUP FAILED", fill="#ffb5af"
            )
            canvas.itemconfigure.assert_any_call(
                canvas.create_text.return_value, text="Could not start Photo Search.\nfailed", fill="#ffd1cc"
            )

    def test_splash_preserves_status_highlights_and_animates_until_failure(self):
        root, canvas = MagicMock(), MagicMock()
        messages = ('{"kind":"status","text":"Database initialized."}\n'
                    '{"kind":"status","text":"Using CPU"}\n'
                    '{"kind":"status","text":"Loading image search model..."}\n')
        with patch.object(startup_splash.tk, "Tk", return_value=root), \
             patch.object(startup_splash.tk, "Canvas", return_value=canvas), \
             patch.object(startup_splash.tk, "Button"), \
             patch.object(startup_splash, "load_logo"), \
             patch.object(startup_splash.sys, "stdin", io.StringIO(messages)), \
             patch.object(startup_splash.threading, "Thread") as thread:
            thread.side_effect = lambda target, daemon: MagicMock(start=target)
            startup_splash.main()
            animate, update = [call.args[1] for call in root.after.call_args_list[:2]]
            animate()
            canvas.coords.assert_called()
            update()
            canvas.itemconfigure.assert_any_call(
                canvas.create_text.return_value,
                text="Database initialized.\nUsing CPU\nLoading image search model...",
            )
            canvas.itemconfigure.assert_any_call(
                canvas.create_text.return_value,
                text="Could not start Photo Search.\nThe application stopped before startup completed.",
                fill="#ffd1cc",
            )
            canvas.coords.reset_mock()
            animate()
            canvas.coords.assert_not_called()

    def test_splash_closes_on_close_message(self):
        root = MagicMock()
        with patch.object(startup_splash.tk, "Tk", return_value=root), \
             patch.object(startup_splash.tk, "Canvas"), \
             patch.object(startup_splash.tk, "Button"), \
             patch.object(startup_splash, "load_logo"), \
             patch.object(startup_splash.sys, "stdin", io.StringIO('{"kind":"close"}\n')), \
             patch.object(startup_splash.threading, "Thread") as thread:
            thread.side_effect = lambda target, daemon: MagicMock(start=target)
            startup_splash.main()
            update = root.after.call_args.args[1]
            update()
            root.destroy.assert_called_once()

    def test_logo_loads_project_image_independent_of_working_directory(self):
        from PIL import ImageTk
        with patch.object(ImageTk, "PhotoImage") as image:
            root = MagicMock()
            logo = startup_splash.load_logo(root)
            self.assertEqual(startup_splash.LOGO_PATH, Path(__file__).resolve().parents[1] / "images" / "photo_search.jpg")
            self.assertEqual(image.call_args.args[0].size, (42, 42))
            self.assertEqual(image.call_args.kwargs["master"], root)
            self.assertIs(logo, image.return_value)

    def test_missing_logo_is_reported_without_preventing_splash(self):
        root = MagicMock()
        with patch.object(startup_splash.tk, "Tk", return_value=root), \
             patch.object(startup_splash.tk, "Canvas") as canvas, \
             patch.object(startup_splash.tk, "Button"), \
             patch.object(startup_splash, "load_logo", side_effect=OSError("missing")), \
             patch.object(startup_splash.sys, "stderr", new_callable=io.StringIO) as stderr, \
             patch.object(startup_splash.sys, "stdin", io.StringIO('{"kind":"close"}\n')), \
             patch.object(startup_splash.threading, "Thread") as thread:
            thread.side_effect = lambda target, daemon: MagicMock(start=target)
            startup_splash.main()
            self.assertIn("Could not load splash image", stderr.getvalue())
            canvas.return_value.create_image.assert_not_called()
            root.after.call_args.args[1]()
            root.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
