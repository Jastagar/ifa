"""Tests for ifa.tools.memory (remember_fact tool + load_facts helper).

Run: python -m unittest ifa.tests.test_tools_memory -v
"""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock

from ifa.core.context import AgentContext
from ifa.services.db import init_db
from ifa.tools import registry


def _setup_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ifa_test_facts_")
    os.close(fd)
    init_db(path)
    return path


def _ctx(db_path: str) -> AgentContext:
    return AgentContext(tts=MagicMock(), db_path=db_path, n8n_config={})


class RememberFactToolTests(unittest.TestCase):
    def setUp(self):
        registry.clear()
        from ifa.tools.memory import TOOL
        registry.register(TOOL)
        self.db_path = _setup_db()

    def tearDown(self):
        registry.clear()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_registered(self):
        self.assertIsNotNone(registry.get("remember_fact"))

    def test_happy_path_inserts_row(self):
        result = registry.dispatch(
            "remember_fact",
            {"fact": "user's cat is named Luna"},
            _ctx(self.db_path),
        )
        self.assertIn("remember", result.lower())
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT fact FROM facts").fetchall()
        conn.close()
        self.assertEqual(rows, [("user's cat is named Luna",)])

    def test_duplicate_fact_silently_ignored(self):
        ctx = _ctx(self.db_path)
        registry.dispatch("remember_fact", {"fact": "same"}, ctx)
        result = registry.dispatch("remember_fact", {"fact": "same"}, ctx)
        # Handler still returns confirmation
        self.assertIn("remember", result.lower())
        # But only one row
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT fact FROM facts").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)

    def test_empty_fact_rejected_by_schema(self):
        result = registry.dispatch("remember_fact", {"fact": ""}, _ctx(self.db_path))
        self.assertTrue(result.startswith("ERROR"))

    def test_whitespace_only_fact_rejected_by_handler(self):
        """Schema allows non-empty, but handler strips + re-checks."""
        result = registry.dispatch("remember_fact", {"fact": "   "}, _ctx(self.db_path))
        self.assertIn("didn't catch", result.lower())

    def test_missing_fact_rejected(self):
        result = registry.dispatch("remember_fact", {}, _ctx(self.db_path))
        self.assertTrue(result.startswith("ERROR"))

    def test_long_fact_truncated(self):
        very_long = "x" * 5000
        registry.dispatch("remember_fact", {"fact": very_long}, _ctx(self.db_path))
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT fact FROM facts").fetchall()
        conn.close()
        self.assertEqual(len(rows[0][0]), 1000)

    def test_extra_args_rejected(self):
        result = registry.dispatch(
            "remember_fact",
            {"fact": "ok", "extra": "bad"},
            _ctx(self.db_path),
        )
        self.assertTrue(result.startswith("ERROR"))


class LoadFactsTests(unittest.TestCase):
    def setUp(self):
        self.db_path = _setup_db()

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_empty_db_returns_empty_list(self):
        from ifa.tools.memory import load_facts
        self.assertEqual(load_facts(self.db_path), [])

    def test_returns_stored_facts(self):
        from ifa.tools.memory import load_facts
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO facts (fact) VALUES (?)", ("fact1",))
        conn.execute("INSERT INTO facts (fact) VALUES (?)", ("fact2",))
        conn.commit()
        conn.close()
        facts = load_facts(self.db_path)
        self.assertEqual(sorted(facts), ["fact1", "fact2"])

    def test_respects_limit(self):
        from ifa.tools.memory import load_facts
        conn = sqlite3.connect(self.db_path)
        for i in range(10):
            conn.execute("INSERT INTO facts (fact) VALUES (?)", (f"fact{i}",))
        conn.commit()
        conn.close()
        facts = load_facts(self.db_path, limit=3)
        self.assertEqual(len(facts), 3)

    def test_missing_db_returns_empty(self):
        from ifa.tools.memory import load_facts
        self.assertEqual(load_facts("/nonexistent/path/db.db"), [])


class AgentSystemPromptWithFactsTests(unittest.TestCase):
    """Integration — facts from DB are injected into system prompt via agent_turn."""

    def setUp(self):
        self.db_path = _setup_db()
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO facts (fact) VALUES (?)", ("user has a cat named Luna",))
        conn.commit()
        conn.close()

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_facts_appear_in_system_prompt(self):
        from unittest.mock import patch
        from ifa.core import agent
        from ifa.core.memory import Memory
        from ifa.tools import registry

        registry.clear()
        ctx = AgentContext(tts=MagicMock(), db_path=self.db_path, n8n_config={})

        with patch("ifa.core.agent.chat", return_value={"message": {"content": "ok"}}) as mock_chat:
            agent.agent_turn("hi", ctx, Memory())

        system_msg = mock_chat.call_args.kwargs["messages"][0]
        self.assertEqual(system_msg["role"], "system")
        self.assertIn("Luna", system_msg["content"])
        self.assertIn("Known facts", system_msg["content"])


if __name__ == "__main__":
    unittest.main()
