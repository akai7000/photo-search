import asyncio
import runpy
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from core import database
from core.face_clustering import cluster_faces


class PeopleSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(self.temp.name) / "test.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()

    def photo(self, name):
        return database.insert_or_update_photo({"path": name})

    def face(self, photo_id, cluster_id, vector=(1.0, 0.0)):
        database.insert_face(photo_id, [0, 0, 10, 10], np.asarray(vector, dtype=np.float32).tobytes())
        conn = database.get_connection()
        try:
            face_id = conn.execute("SELECT MAX(id) FROM faces").fetchone()[0]
            conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster_id, face_id))
            conn.commit()
        finally:
            conn.close()
        return face_id

    def test_naming_and_search_finds_distinct_photos_case_insensitively(self):
        first, second = self.photo("first.jpg"), self.photo("second.jpg")
        self.face(first, 0)
        self.face(first, 0)
        self.face(second, 0)
        database.assign_name_to_cluster(0, "Ren\u00e9e")
        self.assertEqual(set(database.find_photos_by_person_name("  REN\u00c9E  ")), {first, second})
        self.assertIsNone(database.find_photos_by_person_name("unknown"))

    def test_unnamed_groups_listed_first_newest_first(self):
        old, new = self.photo("old.jpg"), self.photo("new.jpg")
        self.face(old, 0)
        self.face(old, 1)
        self.face(new, 2)
        database.assign_name_to_cluster(0, "Old Person")
        order = [(r["person_name"], r["cluster_id"]) for r in database.get_clusters()]
        self.assertEqual(order, [(None, 2), (None, 1), ("Old Person", None)])

    def test_name_inside_sentence_and_group_filter(self):
        solo, pair, crowd, other = (self.photo(n) for n in ("solo", "pair", "crowd", "other"))
        self.face(solo, 0)
        self.face(pair, 0)
        self.face(pair, 1)
        self.face(crowd, 0)
        self.face(crowd, 1)
        self.face(crowd, 2)
        self.face(other, 1)
        database.assign_name_to_cluster(0, "Andrei")
        database.assign_name_to_cluster(1, "Blair")

        group = database.search_people_query("group photo including ANDREI")
        self.assertEqual(group["photo_ids"], [crowd])
        self.assertEqual(group["remaining_text"], "")

        plain = database.search_people_query("photos with andrei")
        self.assertEqual(set(plain["photo_ids"]), {solo, pair, crowd})
        self.assertEqual(plain["photo_ids"][0], crowd)

        both = database.search_people_query("Andrei and Blair")
        self.assertEqual(set(both["photo_ids"]), {pair, crowd})

        scene = database.search_people_query("Andrei at the beach")
        self.assertEqual(scene["remaining_text"], "at beach")
        self.assertIsNone(database.search_people_query("sunset over the beach"))
        self.assertIsNone(database.search_people_query("Andreinov"))

    def test_only_and_exclusion_wording(self):
        pair, crowd, solo, stranger, inna_solo = (
            self.photo(n) for n in ("pair", "crowd", "solo", "stranger", "inna_solo")
        )
        self.face(pair, 0)
        self.face(pair, 1)
        self.face(crowd, 0)
        self.face(crowd, 1)
        self.face(crowd, 2)
        self.face(solo, 0)
        self.face(stranger, 0)
        self.face(stranger, 1)
        self.face(stranger, None)
        self.face(inna_solo, 1)
        database.assign_name_to_cluster(0, "Andrei")
        database.assign_name_to_cluster(1, "Inna")
        search = database.search_people_query

        self.assertEqual(search("inna and andrei only")["photo_ids"], [pair])
        self.assertEqual(set(search("inna and andrei")["photo_ids"]), {pair, crowd, stranger})
        self.assertEqual(search("just Andrei")["photo_ids"], [solo])
        self.assertEqual(set(search("Andrei without Inna")["photo_ids"]), {solo})
        self.assertEqual(set(search("andrei excluding inna")["photo_ids"]), {solo})
        self.assertEqual(search("andrei without inna")["remaining_text"], "")
        self.assertEqual(set(search("Inna not Andrei")["photo_ids"]), {inna_solo})

    def test_person_id_collision_does_not_merge_clusters(self):
        first, second = self.photo("first.jpg"), self.photo("second.jpg")
        self.face(first, 0)
        self.face(second, 1)
        database.assign_name_to_cluster(0, "Alex")
        self.assertEqual(database.find_photos_by_person_name("alex"), [first])
        clusters = database.get_clusters()
        self.assertEqual([row["cluster_id"] for row in clusters if row["person_id"] is None], [1])
        database.assign_name_to_cluster(1, "Blair")
        self.assertEqual(database.find_photos_by_person_name("alex"), [first])
        self.assertEqual(database.find_photos_by_person_name("blair"), [second])

    def test_reclustering_preserves_named_faces(self):
        named = self.photo("named.jpg")
        self.face(named, 0)
        database.assign_name_to_cluster(0, "Alex")
        unnamed = self.photo("unnamed.jpg")
        self.face(unnamed, 5)
        self.face(unnamed, 5)
        self.assertEqual(cluster_faces(), 1)
        self.assertEqual(database.find_photos_by_person_name("Alex"), [named])
        self.assertEqual(next(row["face_count"] for row in database.get_clusters() if row["person_id"] is None), 2)

    def test_legacy_assignments_are_migrated_once(self):
        conn = database.get_connection()
        try:
            conn.execute("ALTER TABLE faces DROP COLUMN cluster_id")
            first = self.photo("first.jpg")
            second = self.photo("second.jpg")
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Alex')")
            conn.execute("INSERT INTO faces (photo_id, person_id) VALUES (?, 1)", (first,))
            conn.execute("INSERT INTO faces (photo_id, person_id) VALUES (?, 0)", (second,))
            conn.commit()
        finally:
            conn.close()
        database.init_db()
        database.init_db()
        self.assertEqual(database.find_photos_by_person_name("alex"), [first])
        self.assertEqual(next(row["cluster_id"] for row in database.get_clusters() if row["person_id"] is None), 0)

    def test_renaming_and_merging_named_people(self):
        first, second = self.photo("first.jpg"), self.photo("second.jpg")
        self.face(first, 0)
        self.face(second, 1)
        database.assign_name_to_cluster(0, "Alex")
        database.assign_name_to_cluster(1, "Blair")
        alex = next(row["person_id"] for row in database.get_clusters() if row["person_name"] == "Alex")
        database.rename_person(alex, "blair")
        self.assertIsNone(database.find_photos_by_person_name("Alex"))
        self.assertEqual(set(database.find_photos_by_person_name("Blair")), {first, second})

    def test_ui_naming_then_search_and_description_fallback(self):
        photo = self.photo("first.jpg")
        conn = database.get_connection()
        try:
            conn.execute("UPDATE photos SET thumbnail_path = ? WHERE id = ?", ("thumb.jpg", photo))
            conn.commit()
        finally:
            conn.close()
        face_id = self.face(photo, 0)
        ui = MagicMock()
        ui.page.side_effect = lambda _: lambda func: func
        ui.input.return_value.classes.return_value = ui.input.return_value
        ui.input.return_value.on.return_value = ui.input.return_value
        ui.input.return_value.value = "Alex"
        ui.input.return_value.props.return_value = ui.input.return_value
        ui.input.return_value.classes.return_value = ui.input.return_value
        ui.input.return_value.value = "Alex"
        ui.checkbox.return_value.value = False
        nicegui = ModuleType("nicegui")
        nicegui.ui = ui
        nicegui.native = MagicMock()
        nicegui.app = SimpleNamespace(native=SimpleNamespace(on=MagicMock()))
        nicegui.run = MagicMock()
        async def io_bound(func, *args):
            return func(*args)
        nicegui.run.io_bound = io_bound
        def run_timer(interval, callback, **kwargs):
            if not kwargs.get("once"):
                return MagicMock()
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

        ui.timer.side_effect = run_timer
        webview = ModuleType("webview")
        scanner = ModuleType("core.scanner")
        scanner.scan_folders = MagicMock()
        clip = ModuleType("core.clip_engine")
        clip.search_photos = MagicMock(return_value=[])
        with patch.dict("sys.modules", {
            "nicegui": nicegui, "webview": webview, "core.scanner": scanner,
            "core.clip_engine": clip,
        }):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"))
        open_people = namespace["open_name_people"]
        globals_ = open_people.__globals__
        globals_["ensure_face_thumbnails"] = MagicMock(return_value=({face_id: Path("face.jpg")}, {}))
        open_people()
        save = next(call.kwargs["on_click"] for call in ui.button.call_args_list if call.args and call.args[0] == "Name whole group")
        globals_["open_name_people"] = MagicMock()
        save()
        ui.button.reset_mock()
        namespace["main_page"]()
        search = next(call.kwargs["on_click"] for call in ui.button.call_args_list if call.args and call.args[0] == "Search")
        search()
        clip.search_photos.assert_not_called()
        self.assertTrue(any("Found 1 results" in str(call) for call in ui.label.call_args_list))
        ui.image.assert_called_with("thumb.jpg")
        preview = next(call.kwargs["on_click"] for call in ui.button.call_args_list
                       if not call.args and "on_click" in call.kwargs)
        globals_["open_photo_preview"] = MagicMock()
        preview()
        globals_["open_photo_preview"].assert_called_once_with("first.jpg")
        ui.input.return_value.value = "beach sunset"
        search()
        clip.search_photos.assert_called_once_with("beach sunset", top_k=30, min_score=0.22)


if __name__ == "__main__":
    unittest.main()
