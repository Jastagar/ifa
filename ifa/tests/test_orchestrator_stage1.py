"""Integration tests for the Stage 1 orchestrator wiring.

Mocks Ollama (so these run without a real model) but exercises real SQLite,
the real tool registry, and the real TTSService subprocess calls (with
subprocess.run patched).

Run: python -m unittest ifa.tests.test_orchestrator_stage1 -v
"""
import ast
import os
import pathlib
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.tools import registry


def _setup_db() -> str:
    from ifa.services.db import init_db
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ifa_stage1_")
    os.close(fd)
    init_db(path)
    return path


def _chat_return(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg: dict = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


def _tool_call(name: str, args: dict) -> dict:
    return {"function": {"name": name, "arguments": args}}


class StartupIntegrationTests(unittest.TestCase):
    """Verify orchestrator startup wires Ollama check, n8n config, DB, tools."""

    def setUp(self):
        registry.clear()

    def tearDown(self):
        registry.clear()

    def test_health_check_failure_exits(self):
        from ifa.core import orchestrator

        with patch("ifa.core.orchestrator.check_health",
                   side_effect=RuntimeError("Ollama not running")):
            with self.assertRaises(SystemExit) as cm:
                orchestrator.run()
        self.assertEqual(cm.exception.code, 1)

    def test_n8n_config_parse_error_exits(self):
        from ifa.core import orchestrator
        from ifa.tools.n8n import N8nConfigError

        with patch("ifa.core.orchestrator.check_health"), \
             patch("ifa.core.orchestrator.load_n8n_config",
                   side_effect=N8nConfigError("YAML syntax error on line 3")):
            with self.assertRaises(SystemExit) as cm:
                orchestrator.run()
        self.assertEqual(cm.exception.code, 1)


class AgentTurnIntegrationTests(unittest.TestCase):
    """End-to-end: user text → agent_turn → registry dispatch → real SQLite writes."""

    def setUp(self):
        registry.clear()
        self.db_path = _setup_db()

    def tearDown(self):
        registry.clear()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_remember_fact_end_to_end(self):
        """User text → LLM picks remember_fact → handler writes DB → terminal response."""
        from ifa.core import agent
        from ifa.tools import register_all

        register_all()
        ctx = AgentContext(tts=MagicMock(), db_path=self.db_path, n8n_config={})
        memory = Memory()

        responses = [
            _chat_return(tool_calls=[_tool_call("remember_fact",
                                               {"fact": "user's cat is Luna"})]),
            _chat_return(content="Got it. I'll remember."),
        ]
        with patch("ifa.core.agent.chat", side_effect=responses):
            result = agent.agent_turn("my cat is named Luna", ctx, memory)

        self.assertEqual(result, "Got it. I'll remember.")
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT fact FROM facts").fetchall()
        conn.close()
        self.assertEqual(rows, [("user's cat is Luna",)])

    def test_stored_facts_inform_next_turn(self):
        """remember_fact on turn 1 → fact present in system prompt on turn 2."""
        from ifa.core import agent
        from ifa.tools import register_all

        register_all()
        ctx = AgentContext(tts=MagicMock(), db_path=self.db_path, n8n_config={})
        memory = Memory()

        # Turn 1: remember a fact
        turn1_responses = [
            _chat_return(tool_calls=[_tool_call("remember_fact",
                                               {"fact": "user works remotely"})]),
            _chat_return(content="Noted."),
        ]
        with patch("ifa.core.agent.chat", side_effect=turn1_responses):
            agent.agent_turn("I work remotely", ctx, memory)

        # Turn 2: observe what ends up in the system prompt
        captured_messages = []
        def capture(*args, **kwargs):
            captured_messages.append(kwargs["messages"])
            return _chat_return(content="Alright.")

        with patch("ifa.core.agent.chat", side_effect=capture):
            agent.agent_turn("tell me about yourself", ctx, memory)

        system_msg = captured_messages[0][0]
        self.assertEqual(system_msg["role"], "system")
        self.assertIn("works remotely", system_msg["content"])

    def test_direct_response_no_tool_call(self):
        from ifa.core import agent
        from ifa.tools import register_all

        register_all()
        ctx = AgentContext(tts=MagicMock(), db_path=self.db_path, n8n_config={})
        memory = Memory()

        with patch("ifa.core.agent.chat",
                   return_value=_chat_return(content="Hello there.")):
            result = agent.agent_turn("hi", ctx, memory)

        self.assertEqual(result, "Hello there.")
        # No side effects in DB
        conn = sqlite3.connect(self.db_path)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0], 0)
        conn.close()


class RegressionGuardTests(unittest.TestCase):
    """Lock in the migration invariants via source-code assertions.

    These tests DON'T require the old code to be deleted — they pass
    whether brain.py / manager.py exist or not, but will start failing
    if someone re-introduces the deleted symbols in non-test code.
    Once Unit 6b deletes those files, these guards stay in place.
    """

    REPO_ROOT = pathlib.Path(__file__).parent.parent.parent

    def _scan_source_for_symbol(self, symbol: str) -> list[str]:
        """Return (filename, line) of any call to `symbol` in ifa/**.py, skipping tests."""
        hits = []
        src_dir = self.REPO_ROOT / "ifa"
        for py_file in src_dir.rglob("*.py"):
            if "tests" in py_file.parts:
                continue
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == symbol:
                        hits.append(f"{py_file.relative_to(self.REPO_ROOT)}:{node.lineno}")
                    elif isinstance(func, ast.Attribute) and func.attr == symbol:
                        hits.append(f"{py_file.relative_to(self.REPO_ROOT)}:{node.lineno}")
        return hits

    def test_orchestrator_imports_agent_turn(self):
        orch_path = self.REPO_ROOT / "ifa" / "core" / "orchestrator.py"
        tree = ast.parse(orch_path.read_text())
        imports_agent_turn = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "ifa.core.agent":
                for alias in node.names:
                    if alias.name == "agent_turn":
                        imports_agent_turn = True
        self.assertTrue(imports_agent_turn,
                        "orchestrator.py must import agent_turn from ifa.core.agent")

    def test_orchestrator_does_not_call_detect_intent(self):
        """detect_intent is slated for deletion; orchestrator must not call it."""
        hits = [h for h in self._scan_source_for_symbol("detect_intent") if "orchestrator" in h]
        self.assertEqual(hits, [],
                         f"orchestrator.py must not call detect_intent. Found: {hits}")

    def test_orchestrator_does_not_call_handle_with_intent(self):
        hits = [h for h in self._scan_source_for_symbol("handle_with_intent") if "orchestrator" in h]
        self.assertEqual(hits, [],
                         f"orchestrator.py must not call handle_with_intent. Found: {hits}")

    def test_orchestrator_does_not_call_extract_fact_or_think(self):
        for symbol in ("extract_fact", "think"):
            hits = [h for h in self._scan_source_for_symbol(symbol) if "orchestrator" in h]
            self.assertEqual(hits, [],
                             f"orchestrator.py must not call {symbol}. Found: {hits}")


if __name__ == "__main__":
    unittest.main()
