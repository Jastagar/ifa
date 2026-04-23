"""Tests for ifa.core.agent.agent_turn.

Run: python -m unittest ifa.tests.test_agent_loop -v
"""
import unittest
from unittest.mock import MagicMock, patch

import httpx

from ifa.core import agent
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.tools import registry
from ifa.tools.registry import Tool


def _make_ctx() -> AgentContext:
    return AgentContext(tts=MagicMock(), db_path=":memory:", n8n_config={})


def _chat_return(content: str | None = None, tool_calls: list | None = None) -> dict:
    """Build the shape chat() returns from Ollama /api/chat."""
    msg: dict = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


def _tool_call(name: str, args: dict) -> dict:
    return {"function": {"name": name, "arguments": args}}


class TerminalResponseTests(unittest.TestCase):
    def setUp(self):
        registry.clear()

    def tearDown(self):
        registry.clear()

    def test_direct_text_response_returned_and_memorized(self):
        ctx = _make_ctx()
        memory = Memory()
        with patch("ifa.core.agent.chat",
                   return_value=_chat_return(content="Hello there.")):
            result = agent.agent_turn("hi", ctx, memory)
        self.assertEqual(result, "Hello there.")
        recent = memory.get_recent(5)
        self.assertEqual(recent[-2]["role"], "user")
        self.assertEqual(recent[-2]["content"], "hi")
        self.assertEqual(recent[-1]["role"], "assistant")
        self.assertEqual(recent[-1]["content"], "Hello there.")

    def test_empty_text_response_handled(self):
        ctx = _make_ctx()
        memory = Memory()
        with patch("ifa.core.agent.chat", return_value=_chat_return(content="")):
            result = agent.agent_turn("hi", ctx, memory)
        self.assertEqual(result, "")

    def test_null_content_becomes_empty_string(self):
        ctx = _make_ctx()
        memory = Memory()
        with patch("ifa.core.agent.chat",
                   return_value={"message": {"content": None}}):
            result = agent.agent_turn("hi", ctx, memory)
        self.assertEqual(result, "")


class ToolCallFlowTests(unittest.TestCase):
    def setUp(self):
        registry.clear()
        registry.register(Tool(
            name="get_time",
            description="Returns current time",
            parameters={"type": "object", "properties": {}},
            handler=lambda args, ctx: "12:00pm",
        ))

    def tearDown(self):
        registry.clear()

    def test_tool_call_then_terminal(self):
        """LLM returns tool call, we dispatch, feed result back, LLM returns terminal."""
        ctx = _make_ctx()
        memory = Memory()
        responses = [
            _chat_return(tool_calls=[_tool_call("get_time", {})]),
            _chat_return(content="The time is 12:00pm."),
        ]
        with patch("ifa.core.agent.chat", side_effect=responses) as mock_chat:
            result = agent.agent_turn("what time is it?", ctx, memory)
        self.assertEqual(result, "The time is 12:00pm.")
        self.assertEqual(mock_chat.call_count, 2)

        # Second chat call should have seen the tool result in messages
        second_call_messages = mock_chat.call_args_list[1].kwargs["messages"]
        roles = [m.get("role") for m in second_call_messages]
        self.assertIn("tool", roles)

    def test_tool_result_includes_nonce_delimiter(self):
        ctx = _make_ctx()
        memory = Memory()
        responses = [
            _chat_return(tool_calls=[_tool_call("get_time", {})]),
            _chat_return(content="done"),
        ]
        with patch("ifa.core.agent.chat", side_effect=responses) as mock_chat:
            agent.agent_turn("what time is it?", ctx, memory)
        second_call_messages = mock_chat.call_args_list[1].kwargs["messages"]
        tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
        self.assertIn("TOOL_RESULT_", tool_msg["content"])
        self.assertIn("_START", tool_msg["content"])
        self.assertIn("_END", tool_msg["content"])
        self.assertIn("12:00pm", tool_msg["content"])

    def test_tool_result_uses_tool_name_field(self):
        """Regression: tool-role messages must use `tool_name` not `name`."""
        ctx = _make_ctx()
        memory = Memory()
        responses = [
            _chat_return(tool_calls=[_tool_call("get_time", {})]),
            _chat_return(content="done"),
        ]
        with patch("ifa.core.agent.chat", side_effect=responses) as mock_chat:
            agent.agent_turn("what time is it?", ctx, memory)
        second_call_messages = mock_chat.call_args_list[1].kwargs["messages"]
        tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
        self.assertIn("tool_name", tool_msg)
        self.assertEqual(tool_msg["tool_name"], "get_time")
        self.assertNotIn("name", tool_msg)

    def test_unknown_tool_returns_error_result_to_llm(self):
        """Unknown tool doesn't crash — returns error string so LLM can recover."""
        ctx = _make_ctx()
        memory = Memory()
        responses = [
            _chat_return(tool_calls=[_tool_call("nonexistent", {})]),
            _chat_return(content="I don't know how to do that."),
        ]
        with patch("ifa.core.agent.chat", side_effect=responses) as mock_chat:
            result = agent.agent_turn("do thing", ctx, memory)
        self.assertEqual(result, "I don't know how to do that.")
        # Second call should include the ERROR: result as a tool-role message
        second_call_messages = mock_chat.call_args_list[1].kwargs["messages"]
        tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
        self.assertIn("ERROR", tool_msg["content"])

    def test_max_iterations_exhausted(self):
        """If LLM emits a tool call after the cap, return the graceful fallback."""
        ctx = _make_ctx()
        memory = Memory()
        # First response: tool call (uses the 1 hop). Second: another tool call — exceeds cap.
        responses = [
            _chat_return(tool_calls=[_tool_call("get_time", {})]),
            _chat_return(tool_calls=[_tool_call("get_time", {})]),
        ]
        with patch("ifa.core.agent.chat", side_effect=responses):
            result = agent.agent_turn("what", ctx, memory)
        self.assertEqual(result, "I couldn't finish that in one step.")


class RetryTests(unittest.TestCase):
    def setUp(self):
        registry.clear()

    def tearDown(self):
        registry.clear()

    def test_malformed_tool_call_retries_once(self):
        """Arguments must be a dict — if they aren't, agent adds a correction message and retries."""
        ctx = _make_ctx()
        memory = Memory()
        registry.register(Tool(
            name="t", description="",
            parameters={"type": "object"},
            handler=lambda a, c: "ok",
        ))
        # 1st: malformed (arguments = string). 2nd (retry): valid. 3rd: terminal.
        responses = [
            _chat_return(tool_calls=[{"function": {"name": "t", "arguments": "not a dict"}}]),
            _chat_return(tool_calls=[_tool_call("t", {})]),
            _chat_return(content="got it"),
        ]
        with patch("ifa.core.agent.chat", side_effect=responses) as mock_chat:
            result = agent.agent_turn("x", ctx, memory)
        self.assertEqual(result, "got it")
        self.assertEqual(mock_chat.call_count, 3)
        # The retry message should be a user-role correction
        second_messages = mock_chat.call_args_list[1].kwargs["messages"]
        last_user = [m for m in second_messages if m.get("role") == "user"]
        self.assertTrue(any("invalid" in m.get("content", "").lower() for m in last_user))

    def test_malformed_after_retry_returns_graceful_fallback(self):
        ctx = _make_ctx()
        memory = Memory()
        bad = _chat_return(tool_calls=[{"function": {"name": "t", "arguments": "bad"}}])
        with patch("ifa.core.agent.chat", side_effect=[bad, bad]):
            result = agent.agent_turn("x", ctx, memory)
        self.assertEqual(result, "I couldn't figure out how to do that.")


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        registry.clear()

    def tearDown(self):
        registry.clear()

    def test_ollama_http_error_returns_graceful_string(self):
        ctx = _make_ctx()
        memory = Memory()
        with patch("ifa.core.agent.chat",
                   side_effect=httpx.ConnectError("connection refused")):
            result = agent.agent_turn("hi", ctx, memory)
        self.assertIn("problem", result.lower())

    def test_missing_message_key_returns_graceful_string(self):
        """Malformed top-level Ollama response (no `message` key) doesn't crash."""
        ctx = _make_ctx()
        memory = Memory()
        with patch("ifa.core.agent.chat", return_value={"unexpected": "shape"}):
            result = agent.agent_turn("hi", ctx, memory)
        self.assertIn("problem", result.lower())


class SystemPromptTests(unittest.TestCase):
    def test_system_prompt_contains_persona(self):
        nonce = "TOOL_RESULT_xyz"
        prompt = agent._build_system_prompt(nonce)
        self.assertIn("Ifa", prompt)
        self.assertIn("concise", prompt)

    def test_system_prompt_embeds_nonce(self):
        nonce = "TOOL_RESULT_abc"
        prompt = agent._build_system_prompt(nonce)
        self.assertIn(f"{nonce}_START", prompt)
        self.assertIn(f"{nonce}_END", prompt)

    def test_system_prompt_instructs_not_to_echo_secrets(self):
        prompt = agent._build_system_prompt("TOOL_RESULT_x")
        self.assertIn("authentication", prompt.lower())

    def test_facts_injected_when_provided(self):
        prompt = agent._build_system_prompt("TOOL_RESULT_x", facts=["user has a cat named Luna"])
        self.assertIn("Luna", prompt)

    def test_no_facts_section_when_empty(self):
        prompt = agent._build_system_prompt("TOOL_RESULT_x")
        self.assertNotIn("Known facts", prompt)


class NonceTests(unittest.TestCase):
    def test_nonce_is_unique_per_call(self):
        nonces = {agent._new_nonce() for _ in range(100)}
        # 100 unique UUIDs
        self.assertEqual(len(nonces), 100)

    def test_nonce_has_prefix(self):
        n = agent._new_nonce()
        self.assertTrue(n.startswith("TOOL_RESULT_"))


if __name__ == "__main__":
    unittest.main()
