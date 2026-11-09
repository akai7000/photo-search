import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core import frozen_startup, startup_splash


class FrozenStartupTests(unittest.TestCase):
    def test_splash_mode_uses_inherited_streams_and_exits(self):
        source, sink = MagicMock(), MagicMock()
        with patch.object(frozen_startup.multiprocessing, "freeze_support") as freeze, \
             patch.object(sys, "argv", ["PhotoSearch.exe", "--photo-search-splash"]), \
             patch.object(sys, "stdin"), patch.object(sys, "stdout"), \
             patch.object(frozen_startup, "_pipe_stream", side_effect=[source, sink]) as pipe, \
             patch.object(startup_splash, "main") as main:
            with self.assertRaises(SystemExit) as result:
                frozen_startup.initialize()
            self.assertEqual(result.exception.code, 0)
            self.assertIs(sys.stdin, source)
            self.assertIs(sys.stdout, sink)
        freeze.assert_called_once()
        self.assertEqual(pipe.call_args_list[0].args, (-10, "r"))
        self.assertEqual(pipe.call_args_list[1].args, (-11, "w"))
        main.assert_called_once()

    def test_application_mode_writes_log_outside_install_directory(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(os.environ, {"LOCALAPPDATA": temp, "NICEGUI_STORAGE_PATH": ""}), \
             patch.object(sys, "argv", ["PhotoSearch.exe"]), \
             patch.object(sys, "stdout"), patch.object(sys, "stderr"), \
             patch.object(frozen_startup.multiprocessing, "freeze_support"):
            frozen_startup.initialize()
            try:
                print("application started")
            finally:
                sys.stdout.close()
            self.assertEqual(
                (Path(temp) / "Photo Search" / "photo-search.log").read_text(encoding="utf-8"),
                "application started\n",
            )

    def test_application_mode_sets_nicegui_storage_to_user_directory(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(os.environ, {"LOCALAPPDATA": temp}, clear=True), \
             patch.object(sys, "argv", ["PhotoSearch.exe"]), \
             patch.object(sys, "stdout"), patch.object(sys, "stderr"), \
             patch.object(frozen_startup.multiprocessing, "freeze_support"):
            frozen_startup.initialize()
            try:
                self.assertEqual(os.environ["NICEGUI_STORAGE_PATH"],
                                 str(Path(temp) / "Photo Search" / "nicegui"))
            finally:
                sys.stdout.close()

    def test_application_mode_requires_local_app_data(self):
        with patch.dict(os.environ, {"LOCALAPPDATA": ""}), \
             patch.object(sys, "argv", ["PhotoSearch.exe"]), \
             patch.object(frozen_startup.multiprocessing, "freeze_support"):
            with self.assertRaisesRegex(RuntimeError, "LOCALAPPDATA"):
                frozen_startup.initialize()


if __name__ == "__main__":
    unittest.main()
