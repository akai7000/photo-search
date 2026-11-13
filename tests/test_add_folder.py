import asyncio
import runpy
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core import database
from core import data_location, face_thumbnails


class AddFolderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        db_patch = patch.object(database, "DB_PATH", Path(self.temp_dir.name) / "test.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)

        self.ui = MagicMock()
        self.switches = {}
        def make_switch(label, **kwargs):
            control = MagicMock()
            self.switches[label] = control
            return control
        self.ui.switch.side_effect = make_switch
        self.dialog = AsyncMock()
        nicegui = ModuleType("nicegui")
        nicegui.ui = self.ui
        nicegui.native = MagicMock()
        nicegui.app = SimpleNamespace(
            native=SimpleNamespace(
                main_window=SimpleNamespace(create_file_dialog=self.dialog),
                on=MagicMock(),
            )
        )
        nicegui.run = MagicMock()
        webview = ModuleType("webview")
        webview.FileDialog = SimpleNamespace(FOLDER="folder")
        scanner = ModuleType("core.scanner")
        scanner.scan_folders = MagicMock()
        clip_engine = ModuleType("core.clip_engine")
        clip_engine.search_photos = MagicMock()
        face_clustering = ModuleType("core.face_clustering")
        face_clustering.cluster_faces = MagicMock()
        modules = {
            "nicegui": nicegui,
            "webview": webview,
            "core.scanner": scanner,
            "core.clip_engine": clip_engine,
            "core.face_clustering": face_clustering,
        }
        with patch.dict("sys.modules", modules):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"))
        page_functions = [call.args[0] for call in self.ui.page.return_value.call_args_list]
        self.main_page = next(page for page in page_functions if page.__name__ == "main_page")
        self.assertEqual(self.ui.run.call_args.kwargs["window_size"], (1300, 900))
        self.callback = namespace["add_folder"]
        self.state = self.callback.__globals__
        self.refresh_folders = self.state["refresh_folders"]
        self.state["refresh_folders"] = MagicMock()

    async def test_add_folder_persists_and_ignores_duplicates(self):
        self.dialog.return_value = ["C:\\Photos"]

        await self.callback()

        self.assertEqual(self.state["selected_folders"], ["C:\\Photos"])
        self.assertEqual(database.get_folders(), ["C:\\Photos"])
        self.state["refresh_folders"].assert_called_once()
        self.ui.notify.assert_called_with("Added: C:\\Photos")

        await self.callback()

        self.assertEqual(database.get_folders(), ["C:\\Photos"])
        self.assertEqual(self.state["selected_folders"], ["C:\\Photos"])
        self.state["refresh_folders"].assert_called_once()
        self.ui.notify.assert_called_with("Folder already added")

    async def test_cancel_does_not_add_folder(self):
        self.dialog.return_value = None

        await self.callback()

        self.assertEqual(database.get_folders(), [])
        self.assertEqual(self.state["selected_folders"], [])
        self.state["refresh_folders"].assert_not_called()

    async def test_database_failure_does_not_change_selected_folders(self):
        self.dialog.return_value = ["C:\\Photos"]
        self.state["add_folder_db"] = MagicMock(side_effect=RuntimeError("database failed"))

        with self.assertRaisesRegex(RuntimeError, "database failed"):
            await self.callback()

        self.assertEqual(self.state["selected_folders"], [])
        self.state["refresh_folders"].assert_not_called()

    async def test_remove_folder_requires_confirmation_and_preserves_originals(self):
        folder = Path(self.temp_dir.name) / "photos"
        folder.mkdir()
        original = folder / "photo.jpg"
        original.write_bytes(b"original")
        database.add_folder(str(folder))
        database.insert_or_update_photo({"path": str(original)})
        self.state["selected_folders"].append(str(folder))
        self.state["refresh_photo_count"] = MagicMock()
        results = MagicMock()
        self.state["search_results_container"] = results
        self.state["remove_folder"](str(folder))
        self.assertEqual(database.get_photo_count(), 1)
        self.assertEqual(database.get_folders(), [str(folder)])
        self.ui.label.assert_any_call(
            "This removes the folder from the list and deletes its indexed photos, faces and generated "
            "thumbnails, except photos also covered by another searchable folder. "
            "Original photo files and saved people will not be deleted."
        )
        confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                       if call.args and call.args[0] == "Remove folder")
        confirm()
        self.assertEqual(database.get_folders(), [])
        self.assertEqual(database.get_photo_count(), 0)
        self.assertTrue(original.exists())
        self.assertEqual(self.state["selected_folders"], [])
        results.clear.assert_called_once()
        self.state["refresh_photo_count"].assert_called_once()

    async def test_remove_folder_rechecks_busy_state_and_data_directory(self):
        folder = str(Path(self.temp_dir.name) / "photos")
        database.add_folder(folder)
        self.state["selected_folders"].append(folder)
        self.state["remove_folder"](folder)
        confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                       if call.args and call.args[0] == "Remove folder")
        self.state["index_progress"]["active"] = True
        confirm()
        self.assertEqual(database.get_folders(), [folder])
        self.state["index_progress"]["active"] = False
        with patch.object(database, "DB_PATH", Path(self.temp_dir.name) / "other.db"):
            confirm()
        self.assertEqual(database.get_folders(), [folder])
        self.ui.notify.assert_called_with(
            "The data directory changed. Reopen the folder confirmation.", type="warning"
        )

    async def test_remove_folder_reports_database_error_without_changing_ui(self):
        folder = str(Path(self.temp_dir.name) / "photos")
        database.add_folder(folder)
        self.state["selected_folders"].append(folder)
        self.state["remove_folder"](folder)
        confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                       if call.args and call.args[0] == "Remove folder")
        with patch.dict(self.state, {"remove_folder_and_index": MagicMock(side_effect=database.sqlite3.Error("disk error"))}):
            confirm()
        self.assertEqual(database.get_folders(), [folder])
        self.assertEqual(self.state["selected_folders"], [folder])
        self.ui.notify.assert_called_with("Could not remove folder: disk error", type="negative")

    async def test_remove_folder_reports_thumbnail_cleanup_failure(self):
        folder = str(Path(self.temp_dir.name) / "photos")
        database.add_folder(folder)
        self.state["selected_folders"].append(folder)
        self.state["refresh_photo_count"] = MagicMock()
        self.state["remove_folder"](folder)
        confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                       if call.args and call.args[0] == "Remove folder")
        def partial_removal(path):
            database.remove_folder_db(path)
            return 0, ["thumbnail locked"]
        with patch.dict(self.state, {"remove_folder_and_index": partial_removal}):
            confirm()
        self.assertEqual(self.state["selected_folders"], [])
        self.ui.notify.assert_called_with(
            "Folder removed, but 1 generated thumbnail(s) could not be removed. "
            "See the application log for details.", type="warning"
        )

    async def test_settings_dialog_saves_each_change_and_closes(self):
        self.state["open_settings"]()
        self.ui.label.return_value.classes.assert_any_call(
            "w-full bg-grey-8 text-grey-2 text-body2 text-bold px-3 py-2"
        )
        self.ui.switch.assert_any_call("Skip Delete Group Confirmation", value=False)
        toggle = self.switches["Skip Delete Group Confirmation"]
        change = toggle.on_value_change.call_args.args[0]
        change(SimpleNamespace(value=True))
        self.assertTrue(database.load_setting("skip_delete_group_confirmation"))
        change(SimpleNamespace(value=False))
        self.assertFalse(database.load_setting("skip_delete_group_confirmation"))
        change(SimpleNamespace(value=True))
        close = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                     if call.kwargs.get("icon") == "close")
        close()
        self.ui.dialog.return_value.__enter__.return_value.close.assert_called_once()
        self.assertFalse(any(call.args and call.args[0] == "Save settings"
                             for call in self.ui.button.call_args_list))
        self.ui.switch.reset_mock()
        self.state["open_settings"]()
        self.ui.switch.assert_any_call("Skip Delete Group Confirmation", value=True)

    async def test_settings_save_failure_is_reported(self):
        self.state["open_settings"]()
        toggle = self.switches["Skip Delete Group Confirmation"]
        change = toggle.on_value_change.call_args.args[0]
        with patch.dict(self.state, {"save_setting": MagicMock(side_effect=database.sqlite3.Error("disk error"))}):
            change(SimpleNamespace(value=True))
        self.ui.notify.assert_called_with("Could not save setting: disk error", type="negative")
        toggle.set_value.assert_called_once_with(False)
        self.assertFalse(database.load_setting("skip_delete_group_confirmation", False))

    async def test_skip_indexed_setting_defaults_to_true_and_persists(self):
        self.state["open_settings"]()
        self.ui.switch.assert_any_call("Skip Previously Indexed Folders", value=True)
        toggle = self.switches["Skip Previously Indexed Folders"]
        change = toggle.on_value_change.call_args.args[0]
        change(SimpleNamespace(value=False))
        self.assertFalse(database.load_setting("skip_indexed_folders", True))
        self.state["open_settings"]()
        self.ui.switch.assert_called_with("Skip Previously Indexed Folders", value=False)
        toggle = self.switches["Skip Previously Indexed Folders"]
        change = toggle.on_value_change.call_args.args[0]
        with patch.dict(self.state, {"save_setting": MagicMock(side_effect=database.sqlite3.Error("disk error"))}):
            change(SimpleNamespace(value=True))
        toggle.set_value.assert_called_once_with(False)
        self.ui.notify.assert_called_with("Could not save setting: disk error", type="negative")
        self.assertFalse(database.load_setting("skip_indexed_folders"))

    async def test_indexing_passes_setting_and_avoids_clustering_when_all_skipped(self):
        self.state["selected_folders"][:] = ["photos"]
        scan = self.state["scan_folders"]
        scan.return_value = {"added": 0, "updated": 0, "skipped": 0, "errors": 0,
                             "cancelled": False, "folders_skipped": 1}
        async def io_bound(func, *args):
            return func(*args)
        self.state["run"].io_bound = io_bound
        self.state["refresh_photo_count"] = MagicMock()
        await self.state["index_folders"]()
        self.assertTrue(scan.call_args.kwargs["skip_indexed_folders"])
        self.state["cluster_faces"].assert_not_called()
        self.assertIn("Folders skipped: 1", self.ui.notify.call_args.args[0])
        database.save_setting("skip_indexed_folders", False)
        scan.return_value["folders_skipped"] = 0
        self.state["cluster_faces"].return_value = 2
        await self.state["index_folders"]()
        self.assertFalse(scan.call_args.kwargs["skip_indexed_folders"])
        self.state["cluster_faces"].assert_called_once()

    async def test_settings_dialog_is_available_while_indexing(self):
        self.main_page()
        settings = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                        if call.args and call.args[0] == "Settings")
        self.state["index_progress"]["active"] = True
        settings()
        self.ui.navigate.to.assert_not_called()
        self.ui.dialog.return_value.__enter__.return_value.open.assert_called_once()

    async def test_switch_data_directory_refreshes_and_persists_selection(self):
        target = Path(self.temp_dir.name) / "other"
        target.mkdir()
        database.prepare_database(target)
        conn = database.get_connection(target / "photo_search.db")
        conn.execute("INSERT INTO folders (path) VALUES (?)", ("other photos",))
        conn.commit()
        conn.close()
        self.dialog.return_value = [str(target)]
        async def io_bound(func, *args):
            return func(*args)
        self.state["run"].io_bound = io_bound
        refresh_count = MagicMock()
        results = MagicMock()
        self.state["refresh_photo_count"] = refresh_count
        self.state["search_results_container"] = results
        with patch.object(data_location, "CONFIG_PATH", Path(self.temp_dir.name) / "config.json"), \
             patch.object(face_thumbnails, "FACE_THUMBNAIL_DIR"):
            self.state["open_settings"]()
            self.ui.input.assert_any_call(label="Data directory", value=str(database.DB_PATH.parent))
            browse = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Browse")
            update = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Update")
            control = self.ui.input.return_value.classes.return_value
            original = database.DB_PATH
            await browse()
            control.set_value.assert_called_once_with(str(target))
            self.assertEqual(database.DB_PATH, original)
            self.assertFalse(data_location.CONFIG_PATH.exists())
            self.state["refresh_folders"].assert_not_called()
            control.value = str(target)
            await update()
            self.assertEqual(database.DB_PATH, target / "photo_search.db")
            self.assertEqual(data_location.load_data_directory(), target)
            self.assertEqual(self.state["selected_folders"], ["other photos"])
            self.assertEqual(self.state["scanner"].THUMBNAIL_DIR, target / "thumbnails")
            self.assertEqual(face_thumbnails.FACE_THUMBNAIL_DIR, target / "face_thumbnails")
            refresh_count.assert_called_once()
            self.state["refresh_folders"].assert_called_once()
            results.clear.assert_called_once()

    async def test_typed_directory_changes_only_after_update(self):
        target = Path(self.temp_dir.name) / "typed"
        target.mkdir()
        async def io_bound(func, *args):
            return func(*args)
        self.state["run"].io_bound = io_bound
        self.state["open_settings"]()
        control = self.ui.input.return_value
        control.props.assert_not_called()
        entry = control.classes.return_value
        entry.value = str(target)
        original = database.DB_PATH
        update = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                      if call.args and call.args[0] == "Update")
        with patch.object(data_location, "CONFIG_PATH", Path(self.temp_dir.name) / "config.json"), \
             patch.object(face_thumbnails, "FACE_THUMBNAIL_DIR"):
            self.assertEqual(database.DB_PATH, original)
            await update()
            self.dialog.assert_not_awaited()
            self.assertEqual(database.DB_PATH, target / "photo_search.db")
            self.assertEqual(data_location.load_data_directory(), target)
            self.state["refresh_folders"].assert_called_once()

    async def test_picker_failure_is_visible(self):
        self.dialog.side_effect = RuntimeError("native picker unavailable")
        with self.assertLogs(level="ERROR"):
            await self.state["choose_data_directory"](MagicMock())
        self.ui.notify.assert_called_with(
            "Could not open folder picker: native picker unavailable",
            type="negative",
        )

    async def test_switch_rejects_corruption_and_busy_operations(self):
        target = Path(self.temp_dir.name) / "bad"
        target.mkdir()
        corrupt = target / "photo_search.db"
        corrupt.write_bytes(b"bad database")
        original = database.DB_PATH
        self.dialog.return_value = [str(target)]
        async def io_bound(func, *args):
            return func(*args)
        self.state["run"].io_bound = io_bound
        settings = MagicMock()
        with patch.object(data_location, "CONFIG_PATH", Path(self.temp_dir.name) / "config.json"):
            await self.state["change_data_directory"](str(target), settings)
            self.assertEqual(database.DB_PATH, original)
            self.assertEqual(corrupt.read_bytes(), b"bad database")
            self.assertFalse(data_location.CONFIG_PATH.exists())
            settings.close.assert_not_called()
            self.assertIn("Could not change data directory", self.ui.notify.call_args.args[0])
            self.dialog.reset_mock()
            for key, value in (("storage_jobs", 1), ("name_people_active", True)):
                self.state[key] = value
                await self.state["change_data_directory"](str(target), settings)
                self.dialog.assert_not_awaited()
                self.state[key] = 0 if key == "storage_jobs" else False

    async def test_switch_creates_empty_index_and_cancel_does_nothing(self):
        target = Path(self.temp_dir.name) / "empty"
        target.mkdir()
        async def io_bound(func, *args):
            return func(*args)
        self.state["run"].io_bound = io_bound
        settings = MagicMock()
        original = database.DB_PATH
        with patch.object(data_location, "CONFIG_PATH", Path(self.temp_dir.name) / "config.json"), \
             patch.object(face_thumbnails, "FACE_THUMBNAIL_DIR"):
            self.dialog.return_value = None
            field = MagicMock()
            field.value = "draft"
            await self.state["choose_data_directory"](field)
            field.set_value.assert_not_called()
            self.assertEqual(field.value, "draft")
            self.assertEqual(database.DB_PATH, original)
            self.assertFalse(data_location.CONFIG_PATH.exists())
            self.dialog.return_value = [str(target)]
            await self.state["change_data_directory"](str(target), settings)
            self.assertTrue((target / "photo_search.db").is_file())
            self.assertEqual(database.get_folders(), [])
            self.assertFalse((target / "thumbnails").exists())
            settings.close.assert_called_once()

    async def test_switch_waits_for_preparation_and_blocks_new_operations(self):
        target = Path(self.temp_dir.name) / "other"
        target.mkdir()
        self.dialog.return_value = [str(target)]
        started = asyncio.Event()
        release = asyncio.Event()
        async def io_bound(func, *args):
            started.set()
            await release.wait()
            return func(*args)
        self.state["run"].io_bound = io_bound
        with patch.object(data_location, "CONFIG_PATH", Path(self.temp_dir.name) / "config.json"), \
             patch.object(face_thumbnails, "FACE_THUMBNAIL_DIR"):
            switch = asyncio.create_task(self.state["change_data_directory"](str(target), MagicMock()))
            await started.wait()
            self.assertTrue(self.state["data_switching"])
            await self.state["index_folders"]()
            self.ui.notify.assert_called_with("Wait for the data directory change to finish.", type="warning")
            release.set()
            await switch
            self.assertFalse(self.state["data_switching"])

    async def test_failed_configuration_save_does_not_activate_new_database(self):
        target = Path(self.temp_dir.name) / "other"
        target.mkdir()
        self.dialog.return_value = [str(target)]
        original = database.DB_PATH
        async def io_bound(func, *args):
            return func(*args)
        self.state["run"].io_bound = io_bound
        settings = MagicMock()
        with patch.dict(self.state, {"save_data_directory": MagicMock(side_effect=OSError("read only"))}):
            await self.state["change_data_directory"](str(target), settings)
        self.assertEqual(database.DB_PATH, original)
        settings.close.assert_not_called()
        self.assertIn("read only", self.ui.notify.call_args.args[0])

    async def test_stale_reset_confirmation_cannot_reset_another_database(self):
        original = database.DB_PATH
        other = Path(self.temp_dir.name) / "other.db"
        database.init_db(other)
        with patch.object(database, "DB_PATH", other):
            self.state["do_reset"](MagicMock(), MagicMock(), original)
            self.ui.notify.assert_called_with(
                "The data directory changed. Reopen Settings to reset its index.", type="warning"
            )

    async def test_face_settings_persist_and_restore_when_reopened(self):
        controls = [MagicMock(), MagicMock()]
        for control in controls:
            control.classes.return_value = control
        self.ui.select.side_effect = controls
        self.state["open_settings"]()
        self.assertEqual(self.ui.select.call_args_list[0].args[0], [24, 32, 40, 48])
        self.assertEqual(self.ui.select.call_args_list[0].kwargs["value"], 24)
        self.assertEqual(self.ui.select.call_args_list[1].args[0], list(range(1, 11)))
        self.assertEqual(self.ui.select.call_args_list[1].kwargs["value"], 3)
        controls[0].on_value_change.call_args.args[0](SimpleNamespace(value=40))
        controls[1].on_value_change.call_args.args[0](SimpleNamespace(value=10))
        self.assertEqual(database.load_setting("face_page_size"), 40)
        self.assertEqual(database.load_setting("initial_open_groups"), 10)
        self.ui.select.side_effect = [MagicMock(), MagicMock()]
        self.state["open_settings"]()
        self.assertEqual(self.ui.select.call_args_list[2].kwargs["value"], 40)
        self.assertEqual(self.ui.select.call_args_list[3].kwargs["value"], 10)

    async def test_face_settings_have_readable_controls_and_explanations(self):
        controls = [MagicMock(), MagicMock()]
        for control in controls:
            control.classes.return_value = control
        self.ui.select.side_effect = controls
        self.state["open_settings"]()
        for control in controls:
            control.classes.assert_called_once_with("w-64 max-w-full settings-select")
        self.ui.label.assert_any_call(
            "Maximum number of face thumbnails shown per page in each group."
        )
        self.ui.label.assert_any_call(
            "Maximum number of unnamed groups opened automatically in Name People."
        )

    async def test_settings_sections_do_not_shrink_inside_scroll_area(self):
        self.state["open_settings"]()
        classes = [call.args[0] for call in self.ui.column.return_value.classes.call_args_list if call.args]
        self.assertIn("w-full min-h-0 gap-4 overflow-y-auto", classes)
        self.assertEqual(classes.count("w-full shrink-0 gap-0 rounded overflow-hidden bg-grey-9"), 3)

    async def test_face_settings_reject_invalid_values_and_revert_on_save_failure(self):
        controls = [MagicMock(), MagicMock()]
        for control in controls:
            control.classes.return_value = control
        self.ui.select.side_effect = controls
        self.state["open_settings"]()
        page_change = controls[0].on_value_change.call_args.args[0]
        group_change = controls[1].on_value_change.call_args.args[0]
        page_change(SimpleNamespace(value=25))
        group_change(SimpleNamespace(value=11))
        controls[0].set_value.assert_called_once_with(24)
        controls[1].set_value.assert_called_once_with(3)
        self.assertIsNone(database.load_setting("face_page_size"))
        with patch.dict(self.state, {"save_setting": MagicMock(side_effect=database.sqlite3.Error("disk error"))}):
            page_change(SimpleNamespace(value=32))
        controls[0].set_value.assert_called_with(24)
        self.ui.notify.assert_called_with("Could not save setting: disk error", type="negative")
        self.assertIsNone(database.load_setting("face_page_size"))

    async def test_index_folders_uses_database_icon(self):
        self.main_page()
        button = next(call for call in self.ui.button.call_args_list
                      if call.args and call.args[0] == "Index Folders")
        self.assertEqual(button.kwargs["icon"], "storage")

    async def test_header_places_photo_count_after_left_aligned_buttons(self):
        self.main_page()
        self.ui.header.return_value.classes.assert_called_once_with("app-header items-center gap-3 flex-wrap")
        theme_css = self.ui.add_css.call_args.args[0]
        self.assertIn("@layer theme", theme_css)
        self.assertIn(".q-dialog .bg-grey-9", theme_css)
        self.assertIn(".text-primary:not(.bg-primary)", theme_css)
        self.ui.label.return_value.classes.assert_any_call("text-subtitle1 ml-auto")
        button_call = next(call for call in self.ui.method_calls
                           if call[0] == "button" and call.args and call.args[0] == "Settings")
        label_call = next(call for call in self.ui.method_calls
                          if call[0] == "label" and call.args and call.args[0] == "0 photos indexed")
        self.assertLess(self.ui.method_calls.index(button_call), self.ui.method_calls.index(label_call))

    async def test_theme_uses_halved_corner_radii(self):
        self.main_page()
        css = self.ui.add_css.call_args.args[0]
        self.assertRegex(css, r"\.q-card\s*\{[^}]*border-radius:\s*7px;")
        self.assertRegex(css, r"\.folder-expansion\s*\{[^}]*border-radius:\s*7px;")
        self.assertRegex(css, r"\.q-btn\s*\{[^}]*border-radius:\s*4\.5px;")
        self.assertRegex(css, r"\.rounded,[^}]*border-radius:\s*2px;")
        self.assertRegex(css, r"\.q-checkbox__bg\s*\{[^}]*border-radius:\s*1px;")

    async def test_main_page_connects_splash_handoff_to_native_window(self):
        self.main_page()
        gate = self.state["window_ready"]
        self.ui.context.client.on_connect.assert_called_once_with(gate.page_connected)
        self.state["app"].native.on.assert_any_call("shown", gate.window_shown)
        self.state["app"].native.on.assert_any_call("loaded", gate.window_loaded)

    async def test_name_people_shows_spinner_before_preparation_finishes(self):
        release = asyncio.Event()
        data = ([{"cluster_id": 1}], False, 24, 3, [], {}, {})

        async def prepare(*args):
            await release.wait()
            return data

        self.state["run"].io_bound = AsyncMock(side_effect=prepare)
        render = AsyncMock()
        self.state["render_name_people"] = render
        self.state["open_name_people"]()
        loading = self.ui.dialog.return_value.__enter__.return_value
        loading.open.assert_called_once()
        self.ui.spinner.assert_called_once_with(size="lg", color="primary")
        callback = self.ui.timer.call_args.args[1]
        task = asyncio.create_task(callback())
        await asyncio.sleep(0)
        render.assert_not_awaited()
        release.set()
        await task
        render.assert_awaited_once()

    async def test_closing_loading_dialog_prevents_late_name_people_dialog(self):
        release = asyncio.Event()

        async def prepare(*args):
            await release.wait()
            return ([{"cluster_id": 1}], False, 24, 3, [], {}, {})

        self.state["run"].io_bound = AsyncMock(side_effect=prepare)
        render = AsyncMock()
        self.state["render_name_people"] = render
        self.state["open_name_people"]()
        callback = self.ui.timer.call_args.args[1]
        task = asyncio.create_task(callback())
        await asyncio.sleep(0)
        loading = self.ui.dialog.return_value.__enter__.return_value
        loading.on.call_args.args[1]()
        release.set()
        await task
        render.assert_not_awaited()

    async def test_folders_expand_only_when_initially_empty(self):
        self.main_page()
        self.ui.expansion.assert_called_once_with(value=True)
        self.state["selected_folders"].append("C:\\Photos")
        self.ui.expansion.reset_mock()
        self.main_page()
        self.ui.expansion.assert_called_once_with(value=False)

    async def test_folder_summary_updates_without_changing_expansion(self):
        self.state["refresh_folders"] = self.refresh_folders
        self.main_page()
        header = self.state["folders_header"]
        summary = self.state["folders_summary"]
        header.set_text.assert_any_call("Searchable Folders (0)")
        summary.set_text.assert_any_call("0 indexed, 0 partially indexed, 0 not indexed")
        folders = [str(Path(self.temp_dir.name) / name) for name in ("a", "b", "c")]
        for folder in folders:
            database.add_folder(folder)
        database.set_folder_total(folders[0], 2)
        database.insert_or_update_photo({"path": str(Path(folders[0]) / "first.jpg")})
        database.insert_or_update_photo({"path": str(Path(folders[0]) / "subfolder" / "second.jpg")})
        database.mark_folder_indexed(folders[0])
        database.set_folder_total(folders[1], 2)
        database.insert_or_update_photo({"path": str(Path(folders[1]) / "photo.jpg")})
        self.state["selected_folders"][:] = folders
        self.ui.expansion.reset_mock()
        self.refresh_folders()
        header.set_text.assert_any_call("Searchable Folders (3)")
        summary.set_text.assert_called_with("1 indexed, 1 partially indexed, 1 not indexed")
        self.ui.expansion.assert_not_called()
        self.assertTrue(any(call.args and call.args[0] == "Partially indexed (1 of 2)"
                            for call in self.ui.badge.call_args_list))
        self.ui.label.assert_any_call("Indexed files")
        self.ui.label.assert_any_call("2")
        self.ui.label.assert_any_call("1")
        self.ui.label.assert_any_call("-")
        database.remove_folder_db(folders[2])
        self.state["selected_folders"].remove(folders[2])
        self.refresh_folders()
        header.set_text.assert_any_call("Searchable Folders (2)")
        summary.set_text.assert_called_with("1 indexed, 1 partially indexed, 0 not indexed")

    async def test_indexed_date_displays_abbreviated_month_and_unpadded_day(self):
        self.state["refresh_folders"] = self.refresh_folders
        folder = str(Path(self.temp_dir.name) / "photos")
        database.add_folder(folder)
        self.state["selected_folders"].append(folder)
        conn = database.get_connection()
        try:
            from datetime import datetime
            timestamp = datetime(2026, 11, 8, 12).astimezone().isoformat()
            conn.execute("UPDATE folders SET indexed_at = ? WHERE path = ?", (timestamp, folder))
            conn.commit()
        finally:
            conn.close()
        self.main_page()
        self.ui.label.assert_any_call("Nov 8, 2026")

    async def test_main_header_has_no_photo_search_title(self):
        self.main_page()
        self.assertFalse(any(call.args and call.args[0] == "Photo Search"
                             for call in self.ui.label.call_args_list))

    async def test_reset_is_in_settings_with_warning_and_confirmation(self):
        self.main_page()
        self.assertFalse(any(call.args and call.args[0] == "Reset Index"
                             for call in self.ui.button.call_args_list))
        self.state["open_settings"]()
        self.ui.label.assert_any_call("Name People")
        self.ui.label.assert_any_call("Indexing")
        warning = (
            "Deletes indexed photo records, face groups, names, generated thumbnails and settings. "
            "Selected folders and original photos will remain. This cannot be undone."
        )
        self.ui.label.assert_any_call(warning)
        self.ui.label.return_value.classes.assert_any_call("text-red-3")
        reset = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                     if call.args and call.args[0] == "Reset Index")
        self.ui.button.return_value.props.assert_any_call("outline dense no-caps size=12px")
        self.ui.row.return_value.classes.assert_any_call("w-full justify-end")
        reset()
        self.ui.label.assert_any_call("Are you sure you want to reset the index?")
        self.assertTrue(any(call.args and call.args[0] == "Cancel" for call in self.ui.button.call_args_list))
        self.assertTrue(any(call.args and call.args[0] == "Reset" for call in self.ui.button.call_args_list))

    async def test_reset_clears_index_and_settings_but_preserves_originals_and_folders(self):
        folder = Path(self.temp_dir.name) / "photos"
        folder.mkdir()
        original = folder / "photo.jpg"
        original.write_bytes(b"original")
        database.add_folder(str(folder))
        database.insert_or_update_photo({"path": str(original)})
        database.save_setting("face_page_size", 40)
        confirmation, settings = MagicMock(), MagicMock()
        fake_path = MagicMock()
        fake_path.return_value.exists.return_value = False
        with patch.dict(self.state, {"Path": fake_path}):
            self.state["do_reset"](confirmation, settings)
        self.assertEqual(database.get_photo_count(), 0)
        self.assertEqual(database.get_folders(), [str(folder)])
        self.assertEqual(original.read_bytes(), b"original")
        self.assertEqual(database.load_setting("face_page_size", 24), 24)
        confirmation.close.assert_called_once()
        settings.close.assert_called_once()

    async def test_reset_is_blocked_while_indexing_even_if_confirmation_is_open(self):
        self.state["index_progress"]["active"] = True
        settings, confirmation = MagicMock(), MagicMock()
        self.state["confirm_reset"](settings)
        self.ui.dialog.assert_not_called()
        self.state["do_reset"](confirmation, settings)
        confirmation.close.assert_not_called()
        settings.close.assert_not_called()
        self.ui.notify.assert_called_with(
            "Cancel indexing and wait for it to finish before resetting the index.", type="warning"
        )

    async def test_photo_preview_loads_and_displays_bounded_image(self):
        self.state["run"].io_bound = AsyncMock(return_value="data:image/jpeg;base64,preview")
        await self.state["open_photo_preview"]("C:\\Photos\\one.jpg")
        self.state["run"].io_bound.assert_awaited_once_with(
            self.state["create_photo_preview"], "C:\\Photos\\one.jpg"
        )
        self.ui.image.return_value.classes.return_value.props.return_value.set_source.assert_called_once_with(
            "data:image/jpeg;base64,preview"
        )
        self.ui.dialog.return_value.__enter__.return_value.open.assert_called_once()

    async def test_photo_preview_explorer_link_selects_its_original_photo(self):
        first = Path(self.temp_dir.name) / "first folder" / "one photo.jpg"
        second = Path(self.temp_dir.name) / "second folder" / "two photo.jpg"
        for path in (first, second):
            path.parent.mkdir()
            path.write_bytes(b"photo")
        self.state["run"].io_bound = AsyncMock(return_value="data:image/jpeg;base64,preview")
        await self.state["open_photo_preview"](str(first))
        await self.state["open_photo_preview"](str(second))
        links = [call for call in self.ui.button.call_args_list
                 if call.args and call.args[0] in (str(first.parent), str(second.parent))]
        self.assertEqual(len(links), 2)
        self.assertEqual([link.args[0] for link in links], [str(first.parent), str(second.parent)])
        self.ui.row.return_value.classes.assert_any_call("w-full items-center no-wrap")
        self.ui.row.return_value.classes.assert_any_call("min-w-0 flex-1 justify-center no-wrap")
        self.ui.button.return_value.props.return_value.classes.assert_any_call(
            "photo-preview-folder text-blue-3 underline min-w-0 max-w-full"
        )
        self.ui.button.return_value.props.return_value.classes.return_value.tooltip.assert_any_call(
            "Find in File Explorer"
        )
        with patch("subprocess.Popen") as popen:
            for link in links:
                link.kwargs["on_click"]()
        self.assertEqual([call.args[0] for call in popen.call_args_list],
                         [["explorer.exe", "/select,", str(first)],
                          ["explorer.exe", "/select,", str(second)]])

    async def test_photo_preview_reports_missing_source(self):
        self.state["run"].io_bound = AsyncMock(side_effect=FileNotFoundError("missing photo"))
        await self.state["open_photo_preview"]("missing.jpg")
        self.ui.label.return_value.classes.return_value.set_text.assert_called_with(
            "Could not load photo: missing photo"
        )
        link = next(call for call in self.ui.button.call_args_list
                    if call.args and call.args[0] == str(Path("missing.jpg").parent))
        with patch("subprocess.Popen") as popen:
            link.kwargs["on_click"]()
        popen.assert_not_called()
        self.assertIn("Original photo not found:", self.ui.notify.call_args.args[0])

    async def test_closing_photo_preview_before_load_finishes_does_not_update_it(self):
        async def close_during_load(*args):
            dialog = self.ui.dialog.return_value.__enter__.return_value
            dialog.on.call_args.args[1]()
            return "data:image/jpeg;base64,preview"

        self.state["run"].io_bound = AsyncMock(side_effect=close_during_load)
        await self.state["open_photo_preview"]("photo.jpg")
        self.ui.image.return_value.classes.return_value.props.return_value.set_source.assert_not_called()
        self.ui.dialog.return_value.__enter__.return_value.delete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
