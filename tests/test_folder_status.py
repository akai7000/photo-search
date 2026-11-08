import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database


class FolderStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(database, "DB_PATH", Path(self.tmp.name) / "t.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        database.init_db()
        self.folder = os.path.join(self.tmp.name, "pics")
        database.add_folder(self.folder)

    def add_photo(self, name):
        database.insert_or_update_photo({
            "path": os.path.join(self.folder, name), "file_hash": name, "file_size": 1,
            "modified_time": 1.0, "date_taken": None, "width": 1, "height": 1,
            "thumbnail_path": None, "clip_embedding": None,
        })

    def test_not_partial_complete(self):
        self.assertEqual(database.get_folder_status(self.folder), "Not indexed")
        database.set_folder_total(self.folder, 2)
        self.add_photo("a.jpg")
        self.assertEqual(database.get_folder_status(self.folder), "Partially indexed (1 of 2)")
        self.add_photo("b.jpg")
        self.assertEqual(database.get_folder_status(self.folder), "Indexed")


if __name__ == "__main__":
    unittest.main()
