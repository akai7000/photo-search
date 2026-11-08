import json
import itertools
import unittest
from unittest.mock import MagicMock, patch

from core import startup


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


if __name__ == "__main__":
    unittest.main()
