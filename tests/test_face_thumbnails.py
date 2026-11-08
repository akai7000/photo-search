import asyncio
import runpy
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
from PIL import Image

from core import database


class FaceThumbnailTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        db_patch = patch.object(database, "DB_PATH", self.root / "test.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)

        self.ui = MagicMock()
        self.ui.input.return_value.props.return_value = self.ui.input.return_value
        self.ui.input.return_value.classes.return_value = self.ui.input.return_value
        nicegui = ModuleType("nicegui")
        nicegui.ui = self.ui
        nicegui.native = MagicMock()
        nicegui.app = SimpleNamespace(
            native=SimpleNamespace(
                main_window=SimpleNamespace(create_file_dialog=AsyncMock()),
                on=MagicMock(),
            )
        )
        nicegui.run = MagicMock()
        async def io_bound(func, *args):
            return func(*args)

        nicegui.run.io_bound = io_bound
        def run_timer(interval, callback, **kwargs):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                async def run_callbacks():
                    await callback()
                    pending = asyncio.all_tasks() - {asyncio.current_task()}
                    if pending:
                        await asyncio.gather(*pending)
                return asyncio.run(run_callbacks())
            return loop.create_task(callback())

        self.ui.timer.side_effect = run_timer
        webview = ModuleType("webview")
        webview.FileDialog = SimpleNamespace(FOLDER="folder")
        scanner = ModuleType("core.scanner")
        scanner.scan_folders = MagicMock()
        clip = ModuleType("core.clip_engine")
        clip.search_photos = MagicMock()
        clustering = ModuleType("core.face_clustering")
        clustering.cluster_faces = MagicMock()
        with patch.dict("sys.modules", {
            "nicegui": nicegui, "webview": webview, "core.scanner": scanner,
            "core.clip_engine": clip, "core.face_clustering": clustering,
        }):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"))
        self.open_name_people = namespace["open_name_people"]

    def insert_face(self, source, bbox=(0, 0, 20, 20), cluster=0):
        photo_id = database.insert_or_update_photo({"path": str(source)})
        database.insert_face(photo_id, list(bbox), b"embedding")
        conn = database.get_connection()
        try:
            face_id = conn.execute("SELECT MAX(id) FROM faces").fetchone()[0]
            conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster, face_id))
            conn.commit()
        finally:
            conn.close()
        return face_id

    def test_existing_cluster_creates_and_displays_thumbnail(self):
        source = self.root / "photo.jpg"
        image = Image.new("RGB", (80, 40), "blue")
        image.paste("red", (0, 0, 40, 20))
        exif = image.getexif()
        exif[274] = 6
        image.save(source, exif=exif)
        # OpenCV reads the EXIF-corrected image; red is now at the top right.
        decoded = cv2.imread(str(source))
        self.assertEqual(decoded.shape[:2], (80, 40))
        self.assertGreater(int(decoded[10, 30, 2]), 180)
        face_id = self.insert_face(source, (20, 0, 40, 40))

        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()

        displayed = Path(self.ui.image.call_args.args[0])
        self.assertTrue(displayed.is_file())
        self.assertIn(str(face_id), displayed.name)
        with Image.open(displayed) as face:
            red, green, blue = face.getpixel((face.width // 2, face.height // 2))
            self.assertGreater(red, 180)
            self.assertLess(blue, 80)

    def test_missing_source_shows_error_instead_of_blank_row(self):
        self.insert_face(self.root / "missing.jpg")
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
        self.ui.image.assert_not_called()
        self.assertTrue(any("unavailable" in str(call).lower() for call in self.ui.label.call_args_list))
        self.assertTrue(any("thumbnail" in str(call).lower() for call in self.ui.notify.call_args_list))

    def test_hover_previews_and_clicks_use_each_original_photo(self):
        sources = [self.root / "first folder" / "one.jpg", self.root / "second folder" / "two.jpg"]
        previews = []
        for source in sources:
            source.parent.mkdir()
            Image.new("RGB", (50, 50), "red").save(source)
            preview = source.with_name("preview.jpg")
            Image.new("RGB", (30, 30), "blue").save(preview)
            previews.append(preview)
            self.insert_face(source)
            with closing(database.get_connection()) as conn, conn:
                conn.execute("UPDATE photos SET thumbnail_path = ? WHERE path = ?", (str(preview), str(source)))
        images = [MagicMock() for _ in range(4)]
        self.ui.image.side_effect = images
        show_photo = MagicMock()
        self.open_name_people.__globals__["show_photo_in_explorer"] = show_photo

        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()

        self.assertEqual([self.ui.image.call_args_list[i].args[0] for i in (1, 3)],
                         [str(preview) for preview in previews])
        for source in sources:
            button = next(call for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == str(source.parent))
            button.kwargs["on_click"]()
        for image in images[::2]:
            control = image.classes.return_value.props.return_value
            self.assertEqual(control.on.call_args.args[0], "mouseenter")
        self.assertEqual([call.args[0] for call in show_photo.call_args_list],
                         [str(source) for source in sources])
        self.assertFalse(any(call.kwargs.get("icon") == "folder_open"
                             for call in self.ui.button.call_args_list))

    def test_explorer_selects_original_and_reports_missing_or_launch_failure(self):
        source = self.root / "photo with spaces.jpg"
        Image.new("RGB", (20, 20), "red").save(source)
        show_photo = self.open_name_people.__globals__["show_photo_in_explorer"]
        with patch("subprocess.Popen") as popen:
            show_photo(str(source))
            popen.assert_called_once_with(["explorer.exe", "/select,", str(source)])
            show_photo(str(self.root / "missing.jpg"))
            self.assertEqual(popen.call_count, 1)
            self.ui.notify.assert_called_with(
                f"Original photo not found: {self.root / 'missing.jpg'}", type="warning"
            )
            popen.side_effect = OSError("Explorer failed")
            show_photo(str(source))
            self.ui.notify.assert_called_with(
                "Could not open Windows Explorer: Explorer failed", type="negative"
            )

    def test_group_pages_limit_rendering_and_preserve_drafts_and_selections(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(21):
            self.insert_face(source, cluster=cluster)

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            return result

        self.ui.expansion.side_effect = expansion
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            self.assertEqual(self.ui.expansion.call_count, 20)
            self.ui.checkbox.call_args_list[0].kwargs["on_change"](SimpleNamespace(value=True))
            self.ui.input.return_value.on_value_change.call_args_list[0].args[0](SimpleNamespace(value="Draft"))
            change_page = self.ui.pagination.call_args.kwargs["on_change"]
            change_page(SimpleNamespace(value=2))
            self.assertEqual(self.ui.expansion.call_count, 21)
            self.ui.checkbox.reset_mock()
            change_page(SimpleNamespace(value=1))
            self.assertEqual(self.ui.expansion.call_count, 41)
            self.assertTrue(self.ui.checkbox.call_args_list[0].kwargs["value"])
            self.assertEqual(self.ui.input.call_args_list[-20].kwargs["value"], "Draft")

    def test_naming_refreshes_in_place_preserving_other_group_draft_and_selection(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(21):
            self.insert_face(source, cluster=cluster)

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            return result

        self.ui.expansion.side_effect = expansion
        self.ui.checkbox.side_effect = lambda *args, **kwargs: MagicMock(value=kwargs.get("value", False))
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            self.ui.checkbox.call_args_list[1].kwargs["on_change"](SimpleNamespace(value=True))
            self.ui.input.return_value.on_value_change.call_args_list[1].args[0](SimpleNamespace(value="Other draft"))
            self.ui.input.return_value.value = "Alice"
            previous_checkboxes = self.ui.checkbox.call_count
            save = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                        if call.args and call.args[0] == "Name whole group")
            save()
            self.assertEqual(self.ui.dialog.call_count, 2)
            self.ui.dialog.return_value.close.assert_not_called()
            self.assertEqual(len(database.get_clusters()), 21)
            self.assertEqual(self.ui.input.call_args_list[-20].kwargs["value"], "Other draft")
            self.assertTrue(any(call.kwargs["value"]
                                for call in self.ui.checkbox.call_args_list[previous_checkboxes:]
                                if call.args and call.args[0] == "Select"))
            self.assertIn("Alice", self.ui.input.call_args_list[-20].kwargs["autocomplete"])

    def test_naming_opens_next_unnamed_group_according_to_setting(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(5):
            self.insert_face(source, cluster=cluster)
        self.ui.checkbox.side_effect = lambda *args, **kwargs: MagicMock(value=kwargs.get("value", False))
        self.ui.input.return_value.value = "Alice"

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            result.classes.return_value = result
            result.__enter__.return_value = result
            return result

        self.ui.expansion.side_effect = expansion
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            self.assertEqual([call.kwargs["value"] for call in self.ui.expansion.call_args_list],
                             [True, True, True, False, False])
            self.ui.expansion.reset_mock()
            save = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                        if call.args and call.args[0] == "Name whole group")
            save()
            self.assertEqual([call.kwargs["value"] for call in self.ui.expansion.call_args_list],
                             [True, True, True, False, False])

    def test_partial_naming_keeps_open_group_without_opening_an_extra(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(4):
            self.insert_face(source, cluster=cluster)
            self.insert_face(source, cluster=cluster)
        controls = []

        def checkbox(*args, **kwargs):
            control = MagicMock(value=kwargs.get("value", False))
            controls.append(control)
            return control

        self.ui.checkbox.side_effect = checkbox
        self.ui.input.return_value.value = "Alice"

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            result.classes.return_value = result
            result.__enter__.return_value = result
            return result

        self.ui.expansion.side_effect = expansion
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            controls[0].value = True
            self.ui.expansion.reset_mock()
            save = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                        if call.args and call.args[0] == "Name whole group")
            save()
            self.assertEqual([call.kwargs["value"] for call in self.ui.expansion.call_args_list],
                             [True, True, True, False, False])

    def test_naming_respects_nondefault_open_group_settings(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(12):
            self.insert_face(source, cluster=cluster)
        self.ui.checkbox.side_effect = lambda *args, **kwargs: MagicMock(value=kwargs.get("value", False))

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            result.classes.return_value = result
            result.__enter__.return_value = result
            return result

        self.ui.expansion.side_effect = expansion
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            for count in (1, 10):
                with self.subTest(count=count):
                    database.save_setting("initial_open_groups", count)
                    self.ui.input.return_value.value = f"Person {count}"
                    self.ui.button.reset_mock()
                    self.open_name_people()
                    self.ui.expansion.reset_mock()
                    save = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                                if call.args and call.args[0] == "Name whole group")
                    save()
                    groups = database.get_clusters()
                    open_values = [call.kwargs["value"] for call in self.ui.expansion.call_args_list]
                    self.assertEqual(sum(value for group, value in zip(groups, open_values)
                                         if group["person_id"] is None), count)
                    self.assertFalse(any(value for group, value in zip(groups, open_values)
                                         if group["person_id"] is not None))

    def test_deleting_groups_refills_expanded_unnamed_groups(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(6):
            self.insert_face(source, cluster=cluster)
        self.ui.checkbox.side_effect = lambda *args, **kwargs: MagicMock(value=kwargs.get("value", False))

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            result.classes.return_value = result
            result.__enter__.return_value = result
            return result

        self.ui.expansion.side_effect = expansion
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            for skip in (False, True):
                with self.subTest(skip_confirmation=skip):
                    database.save_setting("skip_delete_group_confirmation", skip)
                    self.ui.button.reset_mock()
                    self.open_name_people()
                    self.ui.expansion.reset_mock()
                    delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                                  if call.args and call.args[0] == "Delete group")
                    delete()
                    if not skip:
                        self.assertEqual(self.ui.expansion.call_count, 0)
                        confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                                       if call.args and call.args[0] == "Delete faces")
                        confirm()
                    values = [call.kwargs["value"] for call in self.ui.expansion.call_args_list]
                    self.assertEqual(values[:3], [True, True, True])
                    self.assertEqual(sum(values), 3)

    def test_deleting_selected_faces_refills_only_when_group_disappears(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(5):
            self.insert_face(source, cluster=cluster)
        controls = []

        def checkbox(*args, **kwargs):
            control = MagicMock(value=kwargs.get("value", False))
            controls.append(control)
            return control

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            result.classes.return_value = result
            result.__enter__.return_value = result
            return result

        self.ui.checkbox.side_effect = checkbox
        self.ui.expansion.side_effect = expansion
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            controls[0].value = True
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete selected")
            delete()
            self.ui.expansion.reset_mock()
            confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                           if call.args and call.args[0] == "Delete faces")
            confirm()
            values = [call.kwargs["value"] for call in self.ui.expansion.call_args_list]
            self.assertEqual(values, [True, True, True, False])

    def test_partial_delete_does_not_open_an_extra_group(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(5):
            self.insert_face(source, cluster=cluster)
            self.insert_face(source, cluster=cluster)
        controls = []

        def checkbox(*args, **kwargs):
            control = MagicMock(value=kwargs.get("value", False))
            controls.append(control)
            return control

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            result.classes.return_value = result
            result.__enter__.return_value = result
            return result

        self.ui.checkbox.side_effect = checkbox
        self.ui.expansion.side_effect = expansion
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            controls[0].value = True
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete selected")
            delete()
            self.ui.expansion.reset_mock()
            confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                           if call.args and call.args[0] == "Delete faces")
            confirm()
            self.assertEqual([call.kwargs["value"] for call in self.ui.expansion.call_args_list],
                             [True, True, True, False, False])

    def test_refresh_failure_keeps_dialog_open_and_can_retry(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source)
        self.ui.input.return_value.value = "Alice"
        self.ui.checkbox.return_value.value = False
        state = self.open_name_people.__globals__
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            save = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                        if call.args and call.args[0] == "Name whole group")
            retry = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                         if call.args and call.args[0] == "Retry updating groups")
            prepare = state["prepare_name_people"]
            state["prepare_name_people"] = MagicMock(side_effect=RuntimeError("database unavailable"))
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                save()
            self.ui.notify.assert_called_with(
                "Could not update groups: database unavailable. Retry updating groups.", type="negative"
            )
            self.ui.dialog.return_value.close.assert_not_called()
            state["prepare_name_people"] = prepare
            retry()
            self.assertEqual(self.ui.dialog.call_count, 2)
            self.assertIsNotNone(database.find_photos_by_person_name("Alice"))

    def test_group_refresh_shows_centered_spinner_without_status_row(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source)
        self.ui.input.return_value.value = "Alice"
        self.ui.checkbox.return_value.value = False
        overlay = MagicMock()
        overlay.classes.return_value = overlay
        overlay.props.return_value = overlay
        overlay.__enter__.return_value = overlay
        self.ui.element.return_value = overlay
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            overlay.classes.assert_called_with(
                "absolute inset-0 flex items-center justify-center z-10 bg-black/40"
            )
            overlay.props.assert_called_with('role="status" aria-label="Updating groups"')
            self.ui.label.assert_any_call("Group 1 (1 faces)")
            self.ui.label.return_value.classes.assert_any_call("w-full text-subtitle1 text-bold")
            self.assertFalse(any(call.args and call.args[0] == "Updating groups..."
                                 for call in self.ui.label.call_args_list))
            pending = []
            self.ui.timer.side_effect = lambda interval, callback, **kwargs: pending.append(callback)
            save = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                        if call.args and call.args[0] == "Name whole group")
            save()
            self.assertTrue(overlay.set_visibility.call_args.args[0])
            asyncio.run(pending[0]())
            self.assertFalse(overlay.set_visibility.call_args.args[0])

    def test_deleting_last_group_keeps_dialog_open_with_empty_state(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source)
        database.save_setting("skip_delete_group_confirmation", True)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete group")
            delete()
            self.assertEqual(database.get_clusters(), [])
            self.assertEqual(self.ui.dialog.call_count, 2)
            self.ui.dialog.return_value.close.assert_not_called()
            self.ui.label.return_value.classes.return_value.set_visibility.assert_any_call(True)

    def test_group_page_change_ignores_pending_face_load(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for cluster in range(21):
            self.insert_face(source, cluster=cluster)
        expansions = []

        def expansion(*, value):
            result = MagicMock()
            result.value = value
            result.classes.return_value = result
            result.__enter__.return_value = result
            expansions.append(result)
            return result

        self.ui.expansion.side_effect = expansion
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            pending = []
            self.ui.timer.side_effect = lambda interval, callback, **kwargs: pending.append(callback)
            change_page = self.ui.pagination.call_args.kwargs["on_change"]
            change_page(SimpleNamespace(value=2))
            expansions[-1].on_value_change.call_args.args[0](SimpleNamespace(value=True))
            self.assertEqual(len(pending), 1)
            change_page(SimpleNamespace(value=1))
            images_before = self.ui.image.call_count
            asyncio.run(pending[0]())
            self.assertEqual(self.ui.image.call_count, images_before)

    def test_stale_page_results_and_closed_dialog_do_not_render(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for _ in range(25):
            self.insert_face(source)

        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            next_page = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                             if call.kwargs.get("icon") == "arrow_forward")
            previous_page = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                                 if call.kwargs.get("icon") == "arrow_back")
            pending = []
            self.ui.timer.side_effect = lambda interval, callback, **kwargs: pending.append(callback)
            next_page()
            previous_page()
            self.assertEqual(len(pending), 2)

            asyncio.run(pending[1]())
            images_after_latest = self.ui.image.call_count
            asyncio.run(pending[0]())
            self.assertEqual(self.ui.image.call_count, images_after_latest)

            next_page()
            hide_handlers = [call.args[1] for call in self.ui.dialog.return_value.on.call_args_list
                             if call.args[0] == "hide"]
            hide_handlers[-1]()
            asyncio.run(pending[2]())
            self.assertEqual(self.ui.image.call_count, images_after_latest)

    def test_face_folder_and_explorer_on_next_page(self):
        first = self.root / "first.jpg"
        second = self.root / "other folder" / "second.jpg"
        second.parent.mkdir()
        for source in (first, second):
            Image.new("RGB", (50, 50), "red").save(source)
        for _ in range(24):
            self.insert_face(first)
        self.insert_face(second)
        show_photo = MagicMock()
        self.open_name_people.__globals__["show_photo_in_explorer"] = show_photo
        image = MagicMock()
        self.ui.image.return_value = image

        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            next_page = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                             if call.kwargs.get("icon") == "arrow_forward")
            next_page()

        button = next(call for call in self.ui.button.call_args_list
                      if call.args and call.args[0] == str(second.parent))
        button.kwargs["on_click"]()
        show_photo.assert_called_once_with(str(second))

    def test_unavailable_faces_can_be_named_or_deleted_after_confirmation(self):
        self.insert_face(self.root / "missing.jpg")
        self.ui.input.return_value.value = "Alice"
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
        self.open_name_people.__globals__["open_name_people"] = MagicMock()
        save = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                    if call.args and call.args[0] == "Name whole group")
        delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                      if call.args and call.args[0] == "Delete group")
        delete()
        self.ui.label.assert_any_call("Delete all 1 face detections in this group?")
        self.assertEqual(database.get_clusters()[0]["face_count"], 1)
        save()
        self.assertIsNotNone(database.find_photos_by_person_name("Alice"))

    def test_typed_name_is_used_for_whole_row_and_shown_in_header(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source)
        self.ui.input.return_value.value = "Bob"
        self.ui.checkbox.return_value.value = False
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            button = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Name whole group")
            self.open_name_people.__globals__["open_name_people"] = MagicMock()
            button()
        self.assertIsNotNone(database.find_photos_by_person_name("Bob"))

    def test_failed_write_leaves_no_cache_and_retries(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source)
        output = self.root / "faces"
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", output):
            with patch.object(Image.Image, "save", side_effect=OSError("disk full")):
                self.open_name_people()
            self.assertEqual(list(output.iterdir()), [])
            self.open_name_people()
        self.assertTrue(Path(self.ui.image.call_args.args[0]).is_file())

    def test_examples_are_cached_and_recreated_when_removed(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            thumbnail = Path(self.ui.image.call_args.args[0])
            with patch("core.face_thumbnails.Image.open", side_effect=AssertionError("cache missed")):
                self.open_name_people()
            thumbnail.unlink()
            self.open_name_people()
            self.assertTrue(thumbnail.is_file())

    def test_invalid_face_coordinates_are_reported(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source, (80, 80, 100, 100))
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
        self.ui.image.assert_not_called()
        self.assertTrue(any("unavailable" in str(call).lower() for call in self.ui.label.call_args_list))
        self.assertTrue(any("thumbnail" in str(call).lower() for call in self.ui.notify.call_args_list))

    def test_select_sixth_face_and_delete_remaining_row_after_confirmation(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        face_ids = [self.insert_face(source) for _ in range(7)]
        checkboxes = [MagicMock(value=False) for _ in face_ids]
        self.ui.checkbox.side_effect = checkboxes
        self.ui.input.return_value.value = "Alex"
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            self.assertEqual(self.ui.image.call_count, 7)
            self.assertEqual(self.ui.checkbox.call_count, 7)
            checkboxes[5].value = True
            self.assertIsNone(database.find_photos_by_person_name("Alex"))
            select = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Name whole group")
            self.open_name_people.__globals__["open_name_people"] = MagicMock()
            self.ui.checkbox.side_effect = lambda *args, **kwargs: MagicMock(value=kwargs.get("value", False))
            select()
            conn = database.get_connection()
            try:
                self.assertEqual(conn.execute(
                    "SELECT id FROM faces WHERE person_id IS NOT NULL"
                ).fetchone()["id"], face_ids[5])
            finally:
                conn.close()
            # Reopen the real screen with fresh controls after the split.
            self.ui.button.reset_mock()
            self.ui.checkbox.side_effect = lambda *args, **kwargs: MagicMock(value=False)
            self.open_name_people()
            delete_row = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                              if call.args and call.args[0] == "Delete group")
            delete_row()
            self.assertEqual(len(database.get_clusters()), 2)
            confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                           if call.args and call.args[0] == "Delete faces")
            confirm()
            self.assertEqual(database.get_photo_count(), 1)
            self.assertEqual(database.find_photos_by_person_name("Alex"), [1])
            for face_id in face_ids:
                path = self.root / "faces" / f"face_{face_id}_v2.jpg"
                self.assertEqual(path.exists(), face_id == face_ids[5])

    def test_group_confirmation_preference_persists_and_can_be_restored(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source, cluster=0)
        self.insert_face(source, cluster=1)
        self.insert_face(source, cluster=2)
        checkbox = MagicMock(value=False)
        self.ui.checkbox.side_effect = lambda *args, **kwargs: (
            checkbox if args and args[0] == "Do not ask again for group deletion"
            else MagicMock(value=kwargs.get("value", False))
        )
        thumbnail_globals = self.open_name_people.__globals__["delete_face_thumbnails"].__globals__
        with patch.dict(thumbnail_globals, {"FACE_THUMBNAIL_DIR": self.root / "faces"}):
            self.open_name_people()
            self.ui.switch.assert_not_called()
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete group")
            delete()
            self.assertFalse(database.load_setting("skip_delete_group_confirmation", False))
            skip = self.ui.checkbox.call_args
            self.assertEqual(skip.args[0], "Do not ask again for group deletion")
            checkbox.value = True
            self.assertEqual(len(database.get_clusters()), 3)
            self.open_name_people.__globals__["open_name_people"] = MagicMock()
            confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                           if call.args and call.args[0] == "Delete faces")
            confirm()
            self.assertTrue(database.load_setting("skip_delete_group_confirmation"))
            self.assertEqual(database.get_photo_count(), 1)
            self.assertEqual(len(database.get_clusters()), 2)
            self.ui.button.reset_mock()
            self.open_name_people()
            self.assertFalse(any(call.args and call.args[0] == "Restore group delete confirmations"
                                 for call in self.ui.button.call_args_list))
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete group")
            self.ui.button.reset_mock()
            delete()
            self.assertEqual(len(database.get_clusters()), 1)
            self.assertFalse(any(call.args and call.args[0] == "Delete faces"
                                 for call in self.ui.button.call_args_list))
            self.open_name_people.__globals__["open_settings"]()
            self.ui.switch.return_value.on_value_change.call_args.args[0](SimpleNamespace(value=False))
            self.assertFalse(database.load_setting("skip_delete_group_confirmation"))
            self.ui.button.reset_mock()
            self.open_name_people()
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete group")
            delete()
            self.assertTrue(any(call.args and call.args[0] == "Delete faces"
                                for call in self.ui.button.call_args_list))

    def test_large_group_deletes_all_pages_without_confirmation_when_preference_is_set(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for _ in range(25):
            self.insert_face(source)
        database.save_setting("skip_delete_group_confirmation", True)
        thumbnail_globals = self.open_name_people.__globals__["delete_face_thumbnails"].__globals__
        with patch.dict(thumbnail_globals, {"FACE_THUMBNAIL_DIR": self.root / "faces"}):
            self.open_name_people()
            self.open_name_people.__globals__["open_name_people"] = MagicMock()
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete group")
            delete()
        self.assertEqual(database.get_clusters(), [])
        self.assertEqual(database.get_photo_count(), 1)
        self.assertFalse(any(call.args and call.args[0] == "Delete faces"
                             for call in self.ui.button.call_args_list))

    def test_cancel_does_not_save_confirmation_preference(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        self.insert_face(source)
        self.ui.checkbox.side_effect = lambda *args, **kwargs: MagicMock(value=True)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete group")
            delete()
            cancel = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Cancel")
            cancel()
        self.assertFalse(database.load_setting("skip_delete_group_confirmation", False))
        self.assertEqual(len(database.get_clusters()), 1)

    def test_no_groups_with_skipped_confirmations_shows_empty_message(self):
        database.save_setting("skip_delete_group_confirmation", True)
        self.open_name_people()
        self.ui.notify.assert_called_with("No face groups found. Index photos with people first.")
        self.assertTrue(database.load_setting("skip_delete_group_confirmation"))

    def test_delete_selected_requires_confirmation_and_preserves_other_faces(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        face_ids = [self.insert_face(source) for _ in range(3)]
        database.save_setting("skip_delete_group_confirmation", True)
        checkboxes = [MagicMock(value=False) for _ in face_ids]
        self.ui.checkbox.side_effect = checkboxes
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            shown_paths = [Path(call.args[0]) for call in self.ui.image.call_args_list]
            self.assertEqual(len(shown_paths), 3)
            select_delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                                 if call.args and call.args[0] == "Delete selected")
            select_delete()
            self.ui.notify.assert_called_with("Select at least one face to delete", type="warning")
            checkboxes[1].value = True
            select_delete()
            self.assertEqual(database.get_clusters()[0]["face_count"], 3)
            self.assertTrue(any("Delete 1 selected" in str(call) for call in self.ui.label.call_args_list))
            checkboxes[0].value = True
            self.open_name_people.__globals__["open_name_people"] = MagicMock()
            confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                           if call.args and call.args[0] == "Delete faces")
            self.ui.checkbox.side_effect = lambda *args, **kwargs: MagicMock(value=kwargs.get("value", False))
            confirm()
            self.assertEqual(database.get_clusters()[0]["face_count"], 2)
            self.assertEqual(database.get_photo_count(), 1)
            for index, path in enumerate(shown_paths):
                self.assertEqual(path.exists(), index != 1)

    def test_paged_selection_and_existing_name_only_moves_reviewed_faces(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        ids = [self.insert_face(source) for _ in range(25)]
        other = self.root / "other.jpg"
        Image.new("RGB", (50, 50), "blue").save(other)
        existing_face = self.insert_face(other, cluster=1)
        database.assign_name_to_cluster(1, "Alice")
        controls = []

        def checkbox(*args, **kwargs):
            control = MagicMock()
            control.value = kwargs.get("value", False)
            controls.append(control)
            return control

        self.ui.checkbox.side_effect = checkbox
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
            # Only the first 24 in the large group plus the existing person's face.
            self.assertEqual(self.ui.image.call_count, 25)
            self.assertTrue(all(not (self.root / "faces" / f"face_{fid}_v2.jpg").exists()
                                for fid in ids[24:]))
            buttons = {call.args[0]: call.kwargs["on_click"]
                       for call in reversed(self.ui.button.call_args_list) if call.args}
            page_buttons = {call.kwargs["icon"]: call.kwargs["on_click"]
                            for call in self.ui.button.call_args_list if "icon" in call.kwargs}
            self.assertIn("Alice", self.ui.input.call_args_list[0].kwargs["autocomplete"])
            self.assertIn("Alice", [call.args[0] for call in self.ui.menu_item.call_args_list])
            self.ui.input.return_value.value = "Alice"
            select_all = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                              if call.args and call.args[0] == f"Select all {len(ids)}")
            select_all()
            self.assertTrue(all(control.value for control in controls[:24]))
            buttons["Deselect all"]()
            self.assertTrue(all(not control.value for control in controls[:24]))
            buttons["Select all on page"]()
            self.assertTrue(all(control.value for control in controls[:24]))
            buttons["Deselect all on page"]()
            self.assertTrue(all(not control.value for control in controls[:24]))
            buttons["Select all on page"]()
            page_buttons["arrow_forward"]()
            self.assertEqual(self.ui.image.call_count, 26)
            page_two = controls[-1:]
            self.assertTrue(all(not control.value for control in page_two))
            buttons["Select all on page"]()
            page_buttons["arrow_back"]()
            self.assertTrue(all(control.value for control in controls[-24:]))
            buttons["Deselect all on page"]()
            self.open_name_people.__globals__["open_name_people"] = MagicMock()
            buttons["Name whole group"]()
            conn = database.get_connection()
            try:
                assigned = {row["id"] for row in conn.execute(
                    "SELECT id FROM faces WHERE person_id IS NOT NULL"
                )}
            finally:
                conn.close()
            self.assertEqual(assigned, set(ids[24:]) | {existing_face})
            self.assertEqual(next(row["face_count"] for row in database.get_clusters()
                                  if row["cluster_id"] == 0), 24)

    def test_exactly_24_faces_fit_on_one_page(self):
        source = self.root / "photo.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        for _ in range(24):
            self.insert_face(source)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_name_people()
        self.assertEqual(self.ui.image.call_count, 24)
        self.assertEqual(self.ui.checkbox.call_count, 24)
        self.assertFalse(any(call.kwargs.get("icon") == "arrow_forward"
                             for call in self.ui.button.call_args_list))

    def test_saved_page_sizes_control_pagination(self):
        state = self.open_name_people.__globals__
        state["get_clusters"] = MagicMock(return_value=[{
            "cluster_id": 0, "person_id": None, "person_name": None,
            "face_count": 49, "face_ids": ",".join(map(str, range(1, 50))),
        }])
        thumbnails = MagicMock(return_value=({}, []))
        state["ensure_face_thumbnails"] = thumbnails
        state["get_face_photo_details"] = MagicMock(return_value={})
        for size in (24, 32, 40, 48):
            with self.subTest(size=size):
                database.save_setting("face_page_size", size)
                self.ui.button.reset_mock()
                thumbnails.reset_mock()
                self.open_name_people()
                self.assertEqual(thumbnails.call_args_list[0].args[0], list(range(1, size + 1)))
                next_page = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                                 if call.kwargs.get("icon") == "arrow_forward")
                next_page()
                self.assertEqual(thumbnails.call_args_list[1].args[0],
                                 list(range(size + 1, min(size * 2, 49) + 1)))
                next_page()
                if size == 24:
                    self.assertEqual(thumbnails.call_args_list[2].args[0], [49])
                else:
                    self.assertEqual(thumbnails.call_count, 2)

    def test_initial_open_groups_only_opens_requested_unnamed_groups(self):
        state = self.open_name_people.__globals__
        state["get_clusters"] = MagicMock(return_value=[
            {"cluster_id": i, "person_id": None, "person_name": None,
             "face_count": 1, "face_ids": str(i + 1)} for i in range(10)
        ] + [{"cluster_id": None, "person_id": 1, "person_name": "Alice",
              "face_count": 1, "face_ids": "11"}])
        state["ensure_face_thumbnails"] = MagicMock(return_value=({}, []))
        state["get_face_photo_details"] = MagicMock(return_value={})

        def expansion(*, value):
            control = MagicMock()
            control.classes.return_value.__enter__.return_value = control
            control.value = value
            return control

        self.ui.expansion.side_effect = expansion
        for count in (1, 3, 10):
            with self.subTest(count=count):
                database.save_setting("initial_open_groups", count)
                self.ui.expansion.reset_mock()
                state["ensure_face_thumbnails"].reset_mock()
                self.open_name_people()
                self.assertEqual([call.kwargs["value"] for call in self.ui.expansion.call_args_list],
                                 [i < count for i in range(10)] + [False])
                self.assertEqual(state["ensure_face_thumbnails"].call_count, count)

    def test_invalid_face_settings_report_error(self):
        database.save_setting("face_page_size", 0)
        self.open_name_people.__globals__["get_clusters"] = MagicMock(return_value=[{}])
        with self.assertRaisesRegex(ValueError, "Invalid Face Page Size setting"):
            self.open_name_people()
        self.ui.notify.assert_called_with(
            "Could not open Name People: Invalid Face Page Size setting", type="negative"
        )
        self.ui.dialog.assert_called_once()


if __name__ == "__main__":
    unittest.main()
