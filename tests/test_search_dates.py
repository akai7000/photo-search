import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from core import database
from core.search_dates import parse_date_search


class DateSearchTests(unittest.TestCase):
    def test_examples_extract_dates_and_preserve_people_names(self):
        examples = (
            ("John in 2026", date(2026, 1, 1), date(2027, 1, 1), "John"),
            ("Jane before Feb 2020", None, date(2020, 2, 1), "Jane"),
            ("Lara last year", date(2025, 1, 1), date(2026, 1, 1), "Lara"),
            ("Ming after 10/01/2013", date(2013, 10, 2), None, "Ming"),
        )
        for query, start, end, remaining in examples:
            with self.subTest(query=query):
                result = parse_date_search(query, today=date(2026, 11, 10))
                self.assertEqual((result.start, result.end, result.remaining_text),
                                 (start, end, remaining))

    def test_month_ranges_and_multiple_date_constraints(self):
        result = parse_date_search("summer in Feb 2020 after 01/15/2020")
        self.assertEqual(result.start, date(2020, 2, 1))
        self.assertEqual(result.end, date(2020, 3, 1))
        self.assertEqual(result.remaining_text, "summer")

    def test_invalid_numeric_date_is_reported(self):
        with self.assertRaisesRegex(ValueError, "10/32/2013"):
            parse_date_search("Ming after 10/32/2013")

    def test_date_only_query_has_no_visual_search_text(self):
        result = parse_date_search("in 2026")
        self.assertTrue(result.has_filter)
        self.assertEqual(result.remaining_text, "")

    def test_database_date_filter_uses_inclusive_start_and_exclusive_end(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "photos.db"
            with patch.object(database, "DB_PATH", path):
                database.init_db()
                before = database.insert_or_update_photo({
                    "path": "before.jpg", "date_taken": "2019-12-31T23:59:59",
                })
                start = database.insert_or_update_photo({
                    "path": "start.jpg", "date_taken": "2020-01-01T00:00:00",
                })
                last = database.insert_or_update_photo({
                    "path": "last.jpg", "date_taken": "2020-01-31T23:59:59",
                })
                end = database.insert_or_update_photo({
                    "path": "end.jpg", "date_taken": "2020-02-01T00:00:00",
                })
                database.insert_or_update_photo({"path": "unknown.jpg"})

                self.assertEqual(
                    database.get_photo_ids_by_date(date(2020, 1, 1), date(2020, 2, 1)),
                    [last, start],
                )
                self.assertEqual(
                    database.get_photo_ids_by_date(None, date(2020, 1, 1)),
                    [before],
                )
                self.assertEqual(
                    database.get_photo_ids_by_date(date(2020, 2, 1), None),
                    [end],
                )


if __name__ == "__main__":
    unittest.main()
