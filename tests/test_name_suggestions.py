import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import database


def unit(values):
    vector = np.array(values, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).tobytes()


class NameSuggestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(database, "DB_PATH", Path(self.tmp.name) / "t.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        database.init_db()
        conn = database.get_connection()
        conn.execute(
            "INSERT INTO photos (id, path, file_hash, file_size, modified_time) VALUES (1, 'p.jpg', 'h', 1, 1)"
        )
        conn.commit()
        conn.close()

    def add_face(self, embedding, cluster_id):
        conn = database.get_connection()
        conn.execute(
            "INSERT INTO faces (photo_id, cluster_id, bbox_x, bbox_y, bbox_w, bbox_h, embedding) "
            "VALUES (1, ?, 0, 0, 1, 1, ?)",
            (cluster_id, embedding),
        )
        conn.commit()
        conn.close()

    def test_matching_group_is_suggested_and_different_one_is_not(self):
        for _ in range(3):
            self.add_face(unit([1, 0.1, 0]), cluster_id=1)
        database.assign_name_to_cluster(1, "Andrei")
        self.add_face(unit([1, 0.2, 0]), cluster_id=2)
        self.add_face(unit([0, 0, 1]), cluster_id=3)

        suggestions = database.suggest_group_names()

        self.assertEqual(suggestions[2][0], "Andrei")
        self.assertGreater(suggestions[2][1], 0.9)
        self.assertNotIn(3, suggestions)

    def test_no_named_people_gives_no_suggestions(self):
        self.add_face(unit([1, 0, 0]), cluster_id=1)
        self.assertEqual(database.suggest_group_names(), {})


if __name__ == "__main__":
    unittest.main()
