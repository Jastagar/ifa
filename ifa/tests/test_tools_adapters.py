import os
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

from ifa.core.context import AgentContext
from ifa.tools import registry


def _make_ctx(db_path: str = ":memory:") -> AgentContext:
    return AgentContext(tts=MagicMock(), db_path=db_path, n8n_config={})


class GetTimeToolTests(unittest.TestCase):
    def setUp(self):
        registry.clear()
        from ifa.tools.time import TOOL
        registry.register(TOOL)

    def tearDown(self):
        registry.clear()

    def test_registered(self):
        self.assertIsNotNone(registry.get("get_time"))

    def test_returns_time_string(self):
        result = registry.dispatch("get_time", {}, _make_ctx())
        self.assertIn("Current time is", result)

    def test_schema_allows_empty_args(self):
        result = registry.dispatch("get_time", {}, _make_ctx())
        self.assertNotIn("ERROR", result)

    def test_schema_rejects_extra_args(self):
        result = registry.dispatch("get_time", {"bogus": "value"}, _make_ctx())
        self.assertTrue(result.startswith("ERROR"))


class SetReminderToolTests(unittest.TestCase):
    def setUp(self):
        registry.clear()
        from ifa.tools.reminder import TOOL
        registry.register(TOOL)

        fd, self.db_path = tempfile.mkstemp(suffix=".db", prefix="ifa_test_")
        os.close(fd)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE reminders (id INTEGER PRIMARY KEY, task TEXT, trigger_time INTEGER)"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        registry.clear()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_registered(self):
        self.assertIsNotNone(registry.get("set_reminder"))

    def test_happy_path_inserts_row(self):
        result = registry.dispatch(
            "set_reminder",
            {"task": "stretch", "seconds": 60},
            _make_ctx(self.db_path),
        )
        self.assertIn("stretch", result)
        self.assertIn("60", result)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT task FROM reminders").fetchall()
        conn.close()
        self.assertEqual(rows, [("stretch",)])

    def test_schema_rejects_missing_task(self):
        result = registry.dispatch("set_reminder", {"seconds": 60}, _make_ctx())
        self.assertTrue(result.startswith("ERROR"))

    def test_schema_rejects_missing_seconds(self):
        result = registry.dispatch("set_reminder", {"task": "x"}, _make_ctx())
        self.assertTrue(result.startswith("ERROR"))

    def test_schema_rejects_negative_seconds(self):
        result = registry.dispatch(
            "set_reminder",
            {"task": "x", "seconds": -5},
            _make_ctx(),
        )
        self.assertTrue(result.startswith("ERROR"))

    def test_schema_rejects_string_seconds(self):
        result = registry.dispatch(
            "set_reminder",
            {"task": "x", "seconds": "60"},
            _make_ctx(),
        )
        self.assertTrue(result.startswith("ERROR"))

    def test_schema_rejects_extra_fields(self):
        result = registry.dispatch(
            "set_reminder",
            {"task": "x", "seconds": 60, "extra": "stuff"},
            _make_ctx(),
        )
        self.assertTrue(result.startswith("ERROR"))


class ReminderSkillScheduleTests(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db", prefix="ifa_test_")
        os.close(fd)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE reminders (id INTEGER PRIMARY KEY, task TEXT, trigger_time INTEGER)"
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_schedule_inserts_row(self):
        from ifa.skills.reminder import ReminderSkill

        result = ReminderSkill(MagicMock(), db_path=self.db_path).schedule("test task", 30)

        self.assertIn("test task", result)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT task, trigger_time FROM reminders").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "test task")
        self.assertGreater(rows[0][1], int(time.time()))

    def test_schedule_rejects_zero_seconds(self):
        from ifa.skills.reminder import ReminderSkill

        result = ReminderSkill(MagicMock()).schedule("x", 0)
        self.assertIn("greater than zero", result.lower())

    def test_schedule_fires_tts_after_delay(self):
        from ifa.skills.reminder import ReminderSkill

        tts = MagicMock()
        ReminderSkill(tts, db_path=self.db_path).schedule("ping", 1)
        time.sleep(1.5)
        tts.speak.assert_called_once()
        call_arg = tts.speak.call_args[0][0]
        self.assertIn("ping", call_arg)


if __name__ == "__main__":
    unittest.main()
