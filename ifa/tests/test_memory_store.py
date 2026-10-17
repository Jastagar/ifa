import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from ifa.services import memory_store as m
from ifa.tools import memory as tools

DAY = 86400


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "ifa.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_progress_is_replaced_not_appended(self):
        m.remember(self.db, "Sherlock", "Stopped at episode 8.", "progress", now=0)
        saved = m.remember(self.db, "sherlock series", "Stopped at episode 10.", "progress", now=DAY)
        self.assertEqual(saved["previous"], "Stopped at episode 8.")
        rows = m.all_memories(self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["topic"], "Sherlock")
        self.assertEqual(rows[0]["fact"], "Stopped at episode 10.")

    def test_recall_finds_the_topic_from_a_question(self):
        m.remember(self.db, "Sherlock", "Stopped at season 2, episode 8.", "progress")
        m.remember(self.db, "coffee", "Likes it black, no sugar.", "preference")
        m.remember(self.db, "car service", "Due on 12 November at the Tata service centre.", "fact")
        self.assertEqual(m.recall(self.db, "where was I on sherlock?")[0]["topic"], "Sherlock")
        self.assertEqual(m.recall(self.db, "how do I take my coffee")[0]["topic"], "coffee")
        self.assertEqual(m.recall(self.db, "when is the car serviced")[0]["topic"], "car service")
        self.assertEqual(m.recall(self.db, "what about breaking bad"), [])

    def test_different_topics_stay_separate(self):
        m.remember(self.db, "Sherlock", "Episode 8.", "progress")
        m.remember(self.db, "Severance", "Episode 3.", "progress")
        self.assertEqual(len(m.all_memories(self.db)), 2)

    def test_forget(self):
        m.remember(self.db, "Sherlock", "Episode 8.", "progress")
        self.assertEqual(m.forget(self.db, "the sherlock")["topic"], "Sherlock")
        self.assertIsNone(m.forget(self.db, "Sherlock"))

    def test_prompt_context_has_preferences_but_only_topic_names_for_the_rest(self):
        m.remember(self.db, "coffee", "Likes it black.", "preference")
        m.remember(self.db, "gaming time", "Open Steam and turn on the vibe.", "instruction")
        m.remember(self.db, "Sherlock", "Episode 8.", "progress")
        context = m.prompt_context(self.db)
        self.assertIn("coffee: Likes it black.", context)
        self.assertIn("gaming time: Open Steam", context)
        self.assertIn("saved notes on: Sherlock", context)
        self.assertNotIn("Episode 8", context)

    def test_replace_all_edits_adds_and_deletes(self):
        m.remember(self.db, "Sherlock", "Episode 8.", "progress", now=0)
        m.remember(self.db, "coffee", "Black.", "preference", now=0)
        rows = {r["topic"]: r for r in m.all_memories(self.db)}
        items, errors = m.validate([
            {"id": rows["coffee"]["id"], "topic": "Sherlock", "fact": "Black coffee.", "kind": "preference"},
            {"id": rows["Sherlock"]["id"], "topic": "coffee", "fact": "Episode 9.", "kind": "progress"},
            {"topic": "tea", "fact": "Masala chai.", "kind": "preference"},
        ])
        self.assertEqual(errors, {})
        m.replace_all(self.db, items, now=DAY)
        after = {r["topic"]: r for r in m.all_memories(self.db)}
        self.assertEqual(set(after), {"Sherlock", "coffee", "tea"})
        self.assertEqual(after["coffee"]["fact"], "Episode 9.")
        items, _ = m.validate([{"id": after["tea"]["id"], "topic": "tea", "fact": "Masala chai.", "kind": "preference"}])
        m.replace_all(self.db, items)
        self.assertEqual([r["topic"] for r in m.all_memories(self.db)], ["tea"])

    def test_validate_rejects_bad_rows(self):
        _, errors = m.validate([{"topic": "", "fact": "x"}, {"topic": "a", "fact": "x", "kind": "nope"},
                                {"topic": "Tea", "fact": "x"}, {"topic": "the tea", "fact": "y"}])
        self.assertEqual(set(errors), {0, 1, 3})

    def test_ago(self):
        self.assertEqual(m.ago(0, now=3600), "today")
        self.assertEqual(m.ago(0, now=DAY + 5), "yesterday")
        self.assertEqual(m.ago(0, now=3 * DAY), "3 days ago")
        self.assertEqual(m.ago(0, now=21 * DAY), "3 weeks ago")

    def test_imports_the_old_markdown_file_once(self):
        md = pathlib.Path(self.tmp.name) / "memory.md"
        md.write_text("# Memory\n\n## General\n\n- [2026-04-01 10:00] My cat is named Luna\n- I work remotely on Thursdays\n",
                      encoding="utf-8")
        self.assertEqual(m.import_markdown(self.db, md), 2)
        self.assertFalse(md.exists())
        self.assertTrue((md.parent / "memory.md.imported").exists())
        self.assertEqual(m.recall(self.db, "what's my cat called")[0]["fact"], "My cat is named Luna")


class MemoryToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = SimpleNamespace(db_path=os.path.join(self.tmp.name, "ifa.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_sherlock_round_trip(self):
        args = {"topic": "Sherlock", "fact": "Stopped at episode 8.", "kind": "progress"}
        self.assertEqual(tools.TOOL.handler(args, self.ctx), "Saved Sherlock: Stopped at episode 8.")
        result = tools.RECALL_TOOL.handler({"query": "Sherlock"}, self.ctx)
        self.assertIn("Sherlock (progress, saved today): Stopped at episode 8.", result)
        args["fact"] = "Stopped at episode 9."
        self.assertIn("(before: Stopped at episode 8).", tools.TOOL.handler(args, self.ctx))
        self.assertIn("Forgot Sherlock", tools.FORGET_TOOL.handler({"topic": "sherlock"}, self.ctx))
        self.assertIn("Nothing saved", tools.RECALL_TOOL.handler({"query": "Sherlock"}, self.ctx))


if __name__ == "__main__":
    unittest.main()
