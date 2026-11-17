import asyncio
import runpy
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from core import database


class SimpleNamePeopleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        db_patch = patch.object(database, "DB_PATH", self.root / "test.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)

        self.ui = MagicMock()
        self.inputs = []

        def make_input(*args, **kwargs):
            control = MagicMock()
            control.value = kwargs.get("value", "")
            control.props.return_value = control
            control.classes.return_value = control
            self.inputs.append(control)
            return control

        self.ui.input.side_effect = make_input
        nicegui = ModuleType("nicegui")
        nicegui.ui = self.ui
        nicegui.native = MagicMock()
        nicegui.app = SimpleNamespace(native=SimpleNamespace(on=MagicMock()))
        nicegui.run = MagicMock()

        async def io_bound(func, *args):
            return func(*args)

        nicegui.run.io_bound = io_bound

        def timer(interval, callback, **kwargs):
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

        self.ui.timer.side_effect = timer
        modules = {
            "nicegui": nicegui,
            "webview": ModuleType("webview"),
            "core.scanner": ModuleType("core.scanner"),
            "core.clip_engine": ModuleType("core.clip_engine"),
            "core.face_clustering": ModuleType("core.face_clustering"),
        }
        modules["core.scanner"].scan_folders = MagicMock()
        modules["core.clip_engine"].search_photos = MagicMock()
        modules["core.face_clustering"].cluster_faces = MagicMock()
        with patch.dict("sys.modules", modules):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"))
        self.open_people = namespace["open_name_people"]
        self.state = self.open_people.__globals__

    def add_faces(self, cluster, count):
        source = self.root / f"{cluster}.jpg"
        Image.new("RGB", (50, 50), "red").save(source)
        photo_id = database.insert_or_update_photo({"path": str(source)})
        conn = database.get_connection()
        try:
            for _ in range(count):
                conn.execute(
                    "INSERT INTO faces (photo_id, cluster_id, bbox_x, bbox_y, bbox_w, bbox_h) "
                    "VALUES (?, ?, 0, 0, 20, 20)", (photo_id, cluster),
                )
            conn.commit()
        finally:
            conn.close()

    def test_enter_names_group_and_focuses_next_without_checkboxes(self):
        self.add_faces(0, 2)
        self.add_faces(1, 1)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            self.assertEqual(len(self.inputs), 2)
            self.ui.checkbox.assert_not_called()
            self.assertEqual(self.ui.image.call_count, 2)
            first = self.inputs[0]
            first.run_method.assert_called_with("focus")
            first.value = "Alice"
            enter = next(call.args[1] for call in first.on.call_args_list
                         if call.args[0] == "keydown.enter")
            enter()
            self.assertEqual(database.get_people_names(), ["Alice"])
            self.assertEqual(
                [(row["cluster_id"], row["person_name"], row["face_count"])
                 for row in database.get_clusters()],
                [(1, None, 1), (None, "Alice", 2)],
            )
            self.assertEqual(len(self.inputs), 3)
            self.inputs[-1].run_method.assert_called_with("focus")

    def test_review_button_opens_advanced_for_selected_group(self):
        self.add_faces(3, 1)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            review = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.kwargs.get("icon") == "manage_search")
            self.state["open_name_people"] = MagicMock()
            review()
            self.state["open_name_people"].assert_called_once_with(focus_group=(3, None))

    def test_naming_a_middle_group_focuses_the_following_group(self):
        self.add_faces(0, 3)
        self.add_faces(1, 2)
        self.add_faces(2, 1)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            self.inputs[1].value = "Alice"
            enter = next(call.args[1] for call in self.inputs[1].on.call_args_list
                         if call.args[0] == "keydown.enter")
            enter()
            self.assertEqual(len(self.inputs), 5)
            self.inputs[3].run_method.assert_not_called()
            self.inputs[4].run_method.assert_called_with("focus")

    def test_new_prediction_at_same_position_gets_focus_and_enter_accepts_it(self):
        self.add_faces(0, 3)
        self.add_faces(1, 2)
        self.add_faces(2, 1)
        self.state["suggest_group_names"] = MagicMock(side_effect=[
            {}, {2: ("Alice", 0.8)}, {},
        ])
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            self.inputs[0].value = "Alice"
            enter = next(call.args[1] for call in self.inputs[0].on.call_args_list
                         if call.args[0] == "keydown.enter")
            enter()
            predicted = self.inputs[3]
            self.assertEqual(predicted.value, "Alice")
            predicted.run_method.assert_called_with("focus")
            self.inputs[4].run_method.assert_not_called()
            enter = next(call.args[1] for call in predicted.on.call_args_list
                         if call.args[0] == "keydown.enter")
            enter()
        groups = database.get_clusters()
        self.assertEqual(next(row["face_count"] for row in groups if row["person_name"] == "Alice"), 4)
        self.assertEqual(next(row["face_count"] for row in groups if row["cluster_id"] == 1), 2)

    def test_compact_page_shows_40_groups_and_focuses_previous_page_after_last_group(self):
        for cluster in range(41):
            self.add_faces(cluster, 1)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            self.assertEqual(len(self.inputs), 40)
            self.assertEqual(self.ui.pagination.call_args.args[:2], (1, 2))
            self.ui.pagination.call_args.kwargs["on_change"](SimpleNamespace(value=2))
            self.assertEqual(len(self.inputs), 41)
            self.inputs[-1].value = "Alice"
            enter = next(call.args[1] for call in self.inputs[-1].on.call_args_list
                         if call.args[0] == "keydown.enter")
            enter()
            self.assertEqual(len(self.inputs), 81)
            self.inputs[-1].run_method.assert_called_with("focus")

    def test_edited_and_cleared_predictions_survive_rerender(self):
        for cluster in range(3):
            self.add_faces(cluster, 1)
        self.state["suggest_group_names"] = MagicMock(return_value={
            0: ("Alice", 0.8), 1: ("Bob", 0.8), 2: ("Carol", 0.8),
        })
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            predicted = {control.value: control for control in self.inputs}
            predicted["Alice"].on_value_change.call_args.args[0](SimpleNamespace(value="Custom"))
            predicted["Bob"].on_value_change.call_args.args[0](SimpleNamespace(value=""))
            section = self.ui.expansion.return_value.classes.return_value.props.return_value.__enter__.return_value
            section.on_value_change.call_args.args[0](SimpleNamespace(value=True))
            self.assertCountEqual([control.value for control in self.inputs[-3:]], ["Custom", "", "Carol"])

    def test_refreshed_predictions_update_untouched_values_but_not_cleared_drafts(self):
        for cluster, count in ((0, 4), (1, 3), (2, 2), (3, 1)):
            self.add_faces(cluster, count)
        self.state["suggest_group_names"] = MagicMock(side_effect=[
            {2: ("Bob", 0.8), 3: ("Carol", 0.8)},
            {2: ("Dave", 0.8), 3: ("Eve", 0.8)},
        ])
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            self.inputs[0].on_value_change.call_args.args[0](SimpleNamespace(value=""))
            self.inputs[2].value = "Alice"
            enter = next(call.args[1] for call in self.inputs[2].on.call_args_list
                         if call.args[0] == "keydown.enter")
            enter()
            self.assertEqual([control.value for control in self.inputs[-3:]], ["", "Eve", ""])

    def test_retry_after_refresh_failure_retains_position_focus(self):
        for cluster, count in ((0, 3), (1, 2), (2, 1)):
            self.add_faces(cluster, count)
        self.state["suggest_group_names"] = MagicMock(side_effect=[
            {}, RuntimeError("offline"), {2: ("Alice", 0.8)},
        ])
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            self.inputs[0].value = "Alice"
            enter = next(call.args[1] for call in self.inputs[0].on.call_args_list
                         if call.args[0] == "keydown.enter")
            with self.assertRaisesRegex(RuntimeError, "offline"):
                enter()
            retry = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                         if call.args and call.args[0] == "Retry updating groups")
            retry()
            self.assertEqual(self.inputs[3].value, "Alice")
            self.inputs[3].run_method.assert_called_with("focus")

    def test_review_opens_the_page_containing_the_requested_group(self):
        for cluster in range(21):
            self.add_faces(cluster, 1)
        render = AsyncMock()
        self.state["render_name_people"] = render
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(focus_group=(0, None))
        self.assertEqual(render.await_args.args[5:], (2, (0, None)))

    def test_deleting_a_group_requires_confirmation_and_preserves_photos(self):
        self.add_faces(0, 2)
        self.ui.checkbox.return_value.value = False
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.kwargs.get("icon") == "delete")
            delete()
            self.assertEqual(len(database.get_clusters()), 1)
            confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                           if call.args and call.args[0] == "Delete faces")
            confirm()
        self.assertEqual(database.get_clusters(), [])
        self.assertEqual(database.get_photo_count(), 1)

    def test_delete_all_unnamed_requires_confirmation_in_both_views_and_preserves_named_faces(self):
        for simple in (True, False):
            with self.subTest(simple=simple):
                self.ui.reset_mock()
                self.add_faces(0, 2)
                self.add_faces(1, 1)
                database.assign_name_to_cluster(1, "Alice")
                named = next(row for row in database.get_clusters() if row["person_name"] == "Alice")
                named_face_ids = named["face_ids"]
                database.save_setting("skip_delete_group_confirmation", True)
                conn = database.get_connection()
                try:
                    photo = conn.execute("SELECT id FROM photos LIMIT 1").fetchone()[0]
                    conn.executemany(
                        "INSERT INTO faces (photo_id, cluster_id, bbox_x, bbox_y, bbox_w, bbox_h) "
                        "VALUES (?, ?, 0, 0, 20, 20)",
                        [(photo, cluster) for cluster in range(2, 41)],
                    )
                    face_id = conn.execute("SELECT MAX(id) FROM faces").fetchone()[0]
                    conn.commit()
                finally:
                    conn.close()
                thumbnails = self.root / "faces"
                thumbnails.mkdir(exist_ok=True)
                cached = thumbnails / f"face_{face_id}_v2.jpg"
                cached.write_bytes(b"cached face")
                named_cache = thumbnails / f"face_{named_face_ids.split(',')[0]}_v2.jpg"
                named_cache.write_bytes(b"named face")
                with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
                    self.open_people(simple=simple)
                    delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                                  if call.args and call.args[0] == "Delete All Unnamed Groups")
                    delete()
                    self.assertEqual(len(database.get_clusters()), 41)
                    confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                                   if call.args and call.args[0] == "Delete all unnamed groups")
                    confirm()
                remaining = database.get_clusters()
                self.assertEqual([(row["person_name"], row["face_ids"]) for row in remaining],
                                 [("Alice", named_face_ids)])
                self.assertEqual(database.get_people_names(), ["Alice"])
                self.assertEqual(database.get_photo_count(), 2)
                self.assertFalse(cached.exists())
                self.assertTrue(named_cache.exists())

    def test_delete_all_refuses_changes_made_after_confirmation_opens(self):
        self.add_faces(0, 1)
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete All Unnamed Groups")
            delete()
            self.add_faces(1, 1)
            confirm = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                           if call.args and call.args[0] == "Delete all unnamed groups")
            confirm()
        self.assertEqual(len(database.get_clusters()), 2)
        self.assertTrue(any("groups changed" in str(call).lower() for call in self.ui.notify.call_args_list))

    def test_delete_all_unnamed_with_only_named_groups_does_not_open_confirmation(self):
        self.add_faces(0, 1)
        database.assign_name_to_cluster(0, "Alice")
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            delete = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                          if call.args and call.args[0] == "Delete All Unnamed Groups")
            delete()
        self.assertEqual(database.get_people_names(), ["Alice"])
        self.assertFalse(any(call.args and call.args[0] == "Delete all unnamed groups"
                             for call in self.ui.button.call_args_list))
        self.assertTrue(any("no unnamed" in str(call).lower() for call in self.ui.notify.call_args_list))

    def test_named_section_can_rename_a_person(self):
        self.add_faces(0, 2)
        database.assign_name_to_cluster(0, "Alice")
        with patch("core.face_thumbnails.FACE_THUMBNAIL_DIR", self.root / "faces"):
            self.open_people(simple=True)
            section = self.ui.expansion.return_value.classes.return_value.props.return_value.__enter__.return_value
            section.on_value_change.call_args.args[0](SimpleNamespace(value=True))
            self.assertEqual(self.inputs[-1].value, "Alice")
            self.inputs[-1].value = "Bob"
            save = next(call.kwargs["on_click"] for call in self.ui.button.call_args_list
                        if call.args and call.args[0] == "Save")
            save()
        self.assertEqual(database.get_people_names(), ["Bob"])


if __name__ == "__main__":
    unittest.main()
