import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database


class FaceGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(self.temp.name) / "test.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()

    def add_face(self, path, cluster):
        photo = database.insert_or_update_photo({"path": path})
        database.insert_face(photo, [0, 0, 10, 10], b"embedding")
        conn = database.get_connection()
        try:
            face_id = conn.execute("SELECT MAX(id) FROM faces").fetchone()[0]
            conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster, face_id))
            conn.commit()
        finally:
            conn.close()
        return photo, face_id

    def test_move_selected_faces_from_cluster_and_named_person(self):
        a, face_a = self.add_face("a.jpg", 0)
        b, face_b = self.add_face("b.jpg", 0)
        c, face_c = self.add_face("c.jpg", 1)
        database.assign_name_to_cluster(1, "Alice")
        database.assign_selected_faces([face_a], "alice", cluster_id=0, person_id=None)
        self.assertEqual(set(database.find_photos_by_person_name("Alice")), {a, c})
        self.assertEqual(next(row["face_count"] for row in database.get_clusters() if row["cluster_id"] == 0), 1)
        alice = next(row["person_id"] for row in database.get_clusters() if row["person_name"] == "Alice")
        database.assign_selected_faces([face_c], "Bob", cluster_id=None, person_id=alice)
        self.assertEqual(database.find_photos_by_person_name("Alice"), [a])
        self.assertEqual(database.find_photos_by_person_name("Bob"), [c])
        self.assertNotEqual(face_a, face_b)
        self.assertNotIn(b, database.find_photos_by_person_name("Alice"))

    def test_group_representatives_choose_largest_face_per_group(self):
        _, small = self.add_face("small.jpg", 0)
        _, large = self.add_face("large.jpg", 0)
        _, named = self.add_face("named.jpg", 1)
        conn = database.get_connection()
        try:
            conn.execute("UPDATE faces SET bbox_w = 30, bbox_h = 40 WHERE id = ?", (large,))
            conn.commit()
        finally:
            conn.close()
        database.assign_name_to_cluster(1, "Alice")
        person_id = next(row["person_id"] for row in database.get_clusters()
                         if row["person_name"] == "Alice")
        self.assertEqual(database.get_group_representatives(),
                         {(0, None): large, (None, person_id): named})
        self.assertNotEqual(small, large)

    def test_wrong_or_stale_selection_is_atomic(self):
        _, face_a = self.add_face("a.jpg", 0)
        _, face_b = self.add_face("b.jpg", 1)
        with self.assertRaises(ValueError):
            database.assign_selected_faces([face_a, face_b], "Alice", cluster_id=0, person_id=None)
        self.assertIsNone(database.find_photos_by_person_name("Alice"))
        with self.assertRaises(ValueError):
            database.assign_selected_faces([], "Alice", cluster_id=0, person_id=None)
        with self.assertRaises(ValueError):
            database.assign_selected_faces([face_a], " ", cluster_id=0, person_id=None)

    def test_delete_group_preserves_photos_and_other_faces(self):
        a, face_a = self.add_face("a.jpg", 0)
        b, face_b = self.add_face("b.jpg", 0)
        _, face_c = self.add_face("c.jpg", 1)
        with self.assertRaises(ValueError):
            database.delete_face_group([face_a], cluster_id=0, person_id=None)
        self.assertEqual(database.delete_face_group([face_a, face_b], cluster_id=0, person_id=None), [face_a, face_b])
        conn = database.get_connection()
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0], 3)
            self.assertEqual(conn.execute("SELECT id FROM faces").fetchone()[0], face_c)
        finally:
            conn.close()

    def test_deleting_named_row_removes_empty_person(self):
        _, face = self.add_face("a.jpg", 0)
        database.assign_name_to_cluster(0, "Alice")
        person_id = database.get_clusters()[0]["person_id"]
        database.delete_face_group([face], cluster_id=None, person_id=person_id)
        self.assertIsNone(database.find_photos_by_person_name("Alice"))
        self.assertEqual(database.get_clusters(), [])

    def test_delete_all_unnamed_groups_preserves_named_faces_and_ungrouped_faces(self):
        _, first = self.add_face("a.jpg", 0)
        _, named = self.add_face("b.jpg", 1)
        _, ungrouped = self.add_face("c.jpg", None)
        database.assign_name_to_cluster(1, "Alice")
        database.get_or_create_person("Unused")
        conn = database.get_connection()
        try:
            conn.execute("UPDATE faces SET cluster_id = 0 WHERE id = ?", (named,))
            conn.commit()
        finally:
            conn.close()
        groups = [(0, [first])]
        self.assertEqual(database.delete_all_unnamed_face_groups(groups), [first])
        self.assertEqual([row["person_name"] for row in database.get_clusters()], ["Alice"])
        self.assertEqual(database.get_people_names(), ["Alice", "Unused"])
        self.assertEqual(database.get_photo_count(), 3)
        conn = database.get_connection()
        try:
            self.assertEqual({row["id"] for row in conn.execute("SELECT id FROM faces")}, {named, ungrouped})
        finally:
            conn.close()

    def test_delete_all_unnamed_groups_rejects_stale_snapshot_atomically(self):
        _, first = self.add_face("a.jpg", 0)
        groups = [(0, [first])]
        _, second = self.add_face("b.jpg", 1)
        with self.assertRaises(ValueError):
            database.delete_all_unnamed_face_groups(groups)
        self.assertEqual({int(row["face_ids"]) for row in database.get_clusters()}, {first, second})

    def test_delete_all_unnamed_groups_rejects_moved_faces_and_rolls_back_database_failure(self):
        _, first = self.add_face("a.jpg", 0)
        _, second = self.add_face("b.jpg", 1)
        groups = [(0, [first]), (1, [second])]
        database.assign_name_to_cluster(0, "Alice")
        with self.assertRaises(ValueError):
            database.delete_all_unnamed_face_groups(groups)
        groups = [(row["cluster_id"], [int(face_id) for face_id in row["face_ids"].split(",")])
                  for row in database.get_clusters() if row["person_id"] is None]
        conn = database.get_connection()
        try:
            conn.execute(
                "CREATE TRIGGER prevent_delete BEFORE DELETE ON faces "
                "BEGIN SELECT RAISE(ABORT, 'delete failed'); END"
            )
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(sqlite3.IntegrityError):
            database.delete_all_unnamed_face_groups(groups)
        self.assertEqual(len(database.get_clusters()), 2)
        self.assertEqual(database.get_people_names(), ["Alice"])

    def test_new_named_group_does_not_invalidate_unnamed_deletion(self):
        _, first = self.add_face("a.jpg", 0)
        groups = [(0, [first])]
        _, named = self.add_face("b.jpg", 1)
        database.assign_name_to_cluster(1, "Alice")
        with self.assertRaises(ValueError):
            database.delete_all_unnamed_face_groups(groups + [(1, [named])])
        self.assertEqual(database.delete_all_unnamed_face_groups(groups), [first])
        self.assertEqual(database.get_people_names(), ["Alice"])
        self.assertEqual(database.get_clusters()[0]["face_ids"], str(named))

    def test_delete_selected_faces_preserves_other_detections_and_photos(self):
        first_photo, first_face = self.add_face("a.jpg", 0)
        second_photo, second_face = self.add_face("b.jpg", 0)
        _, outside = self.add_face("c.jpg", 1)
        with self.assertRaises(ValueError):
            database.delete_selected_faces([first_face, outside], cluster_id=0, person_id=None)
        with self.assertRaises(ValueError):
            database.delete_selected_faces([], cluster_id=0, person_id=None)
        self.assertEqual(
            database.delete_selected_faces([first_face], cluster_id=0, person_id=None),
            [first_face],
        )
        conn = database.get_connection()
        try:
            self.assertEqual({row["id"] for row in conn.execute("SELECT id FROM faces")}, {second_face, outside})
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0], 3)
        finally:
            conn.close()
        with self.assertRaises(ValueError):
            database.delete_selected_faces([first_face], cluster_id=0, person_id=None)

    def test_delete_selected_named_face_updates_search_and_cleans_empty_person(self):
        first_photo, first_face = self.add_face("a.jpg", 0)
        second_photo, second_face = self.add_face("b.jpg", 0)
        database.assign_name_to_cluster(0, "Alice")
        person_id = database.get_clusters()[0]["person_id"]
        database.delete_selected_faces([first_face], cluster_id=None, person_id=person_id)
        self.assertEqual(database.find_photos_by_person_name("Alice"), [second_photo])
        database.delete_selected_faces([second_face], cluster_id=None, person_id=person_id)
        self.assertIsNone(database.find_photos_by_person_name("Alice"))
        self.assertEqual(database.get_photo_count(), 2)


if __name__ == "__main__":
    unittest.main()
