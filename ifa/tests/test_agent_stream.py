"""Regression tests for the streaming agent loop.

The LLM is replaced by a scripted fake: each test lists the replies the
"model" gives on each call, then asserts what the loop did with them.
This tests the loop's plumbing (hops, budgets, the tool-message protocol)
deterministically — whether the real model *chooses* well is the job of
test_tool_call_bench.py.
"""
import copy
import unittest
from unittest.mock import MagicMock, patch

from ifa.core import agent_stream
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.tools import registry
from ifa.tools.registry import Tool


def _text(content: str) -> list[dict]:
    return [{"message": {"content": content}}]


def _calls(*calls: tuple[str, dict]) -> list[dict]:
    return [{"message": {"content": "", "tool_calls": [
        {"function": {"name": name, "arguments": args}} for name, args in calls
    ]}}]


class ScriptedLLM:
    """Stands in for stream_chat. Replays one scripted reply per call and
    records a snapshot of what the loop sent each time."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, model, messages, tools=None, **_):
        # Snapshot: the loop keeps appending to the same list after this call.
        self.requests.append({"messages": copy.deepcopy(messages), "tools": tools})
        if not self._replies:
            raise AssertionError("LLM called more times than scripted")
        return iter(self._replies.pop(0))


def _tool(name: str, handler) -> Tool:
    return Tool(
        name=name,
        description=f"test tool {name}",
        parameters={"type": "object", "properties": {}, "additionalProperties": True},
        handler=handler,
    )


def _tool_messages(request: dict) -> list[dict]:
    return [m for m in request["messages"] if m["role"] == "tool"]


class AgentLoopTestCase(unittest.TestCase):
    def setUp(self):
        registry.clear()
        self.ctx = AgentContext(tts=MagicMock(), db_path=":memory:", n8n_config={})
        # Keep the system prompt independent of vibe state and ./memory.md.
        for target, value in [
            ("ifa.core.agent_stream.VibeManager.get_system_based_on_vibe", ""),
            ("ifa.core.agent_stream.load_memories", ""),
        ]:
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        registry.clear()

    def run_turn(self, llm: ScriptedLLM, text: str = "hi", spoken: list | None = None) -> str:
        with patch("ifa.core.agent_stream.stream_chat", side_effect=llm):
            return agent_stream.agent_turn_stream(
                text, self.ctx, Memory(), on_sentence=spoken.append if spoken is not None else None
            )


class DirectResponseTests(AgentLoopTestCase):
    def test_direct_response_submits_each_sentence_without_waiting(self):
        """Every available sentence reaches the independent TTS pipeline."""
        spoken = []
        llm = ScriptedLLM([
            {"message": {"content": "Hello there. "}},
            {"message": {"content": "How are you?"}},
        ])

        result = self.run_turn(llm, spoken=spoken)

        self.assertEqual(result, "Hello there. How are you?")
        self.assertEqual(spoken, ["Hello there.", "How are you?"])
        self.assertEqual(len(llm.requests), 1)
        self.assertIsNotNone(llm.requests[0]["tools"])


class MultiHopTests(AgentLoopTestCase):
    def test_second_tool_call_sees_first_tool_result(self):
        """The core of ReAct: act, observe the result, then decide the next act."""
        order = []
        registry.register(_tool("look", lambda a, c: order.append("look") or "the tab says ERROR 404"))
        registry.register(_tool("remind", lambda a, c: order.append("remind") or "reminder set"))
        llm = ScriptedLLM(
            _calls(("look", {})),
            _calls(("remind", {})),
            _text("I set a reminder about the 404."),
        )

        result = self.run_turn(llm)

        self.assertEqual(result, "I set a reminder about the 404.")
        self.assertEqual(order, ["look", "remind"])
        self.assertEqual(len(llm.requests), 3)
        # Hop 2 was decided with hop 1's result in view.
        self.assertIn("ERROR 404", _tool_messages(llm.requests[1])[0]["content"])
        # Tools stay available while under budget.
        self.assertTrue(all(r["tools"] for r in llm.requests))

    def test_tool_error_is_fed_back_so_model_can_recover(self):
        attempts = []

        def flaky(args, ctx):
            attempts.append(args)
            if len(attempts) == 1:
                raise RuntimeError("device offline")
            return "ok"

        registry.register(_tool("play", flaky))
        llm = ScriptedLLM(
            _calls(("play", {})),
            _calls(("play", {"retry": True})),
            _text("Playing now."),
        )

        result = self.run_turn(llm)

        self.assertEqual(result, "Playing now.")
        self.assertEqual(len(attempts), 2)
        self.assertIn("device offline", _tool_messages(llm.requests[1])[0]["content"])


class ToolProtocolTests(AgentLoopTestCase):
    def test_every_tool_call_gets_a_result_even_after_a_failure(self):
        """N tool calls in the assistant message must be answered by N tool messages."""
        ran = []
        registry.register(_tool("broken", lambda a, c: 1 / 0))
        registry.register(_tool("fine", lambda a, c: ran.append("fine") or "fine result"))
        llm = ScriptedLLM(
            _calls(("broken", {}), ("fine", {})),
            _text("One of those failed."),
        )

        self.run_turn(llm)

        results = _tool_messages(llm.requests[1])
        self.assertEqual([m["tool_name"] for m in results], ["broken", "fine"])
        self.assertIn("ERROR", results[0]["content"])
        self.assertIn("fine result", results[1]["content"])
        self.assertEqual(ran, ["fine"])

    def test_unknown_tool_gets_an_error_result_not_a_crash(self):
        llm = ScriptedLLM(_calls(("does_not_exist", {})), _text("Sorry."))

        self.assertEqual(self.run_turn(llm), "Sorry.")
        self.assertIn("unknown tool", _tool_messages(llm.requests[1])[0]["content"])

    def test_results_are_wrapped_in_the_turn_nonce(self):
        registry.register(_tool("t", lambda a, c: "data"))
        llm = ScriptedLLM(_calls(("t", {})), _text("ok"))

        self.run_turn(llm)

        nonce_line = _tool_messages(llm.requests[1])[0]["content"].splitlines()[0]
        system_prompt = llm.requests[1]["messages"][0]["content"]
        nonce = nonce_line.split("_START")[0].lstrip("<")
        self.assertIn(nonce, system_prompt)


class BudgetTests(AgentLoopTestCase):
    def test_final_hop_withholds_tools_so_model_must_answer(self):
        """A model stuck calling tools is forced to answer, not cut off."""
        calls = []
        registry.register(_tool("again", lambda a, c: calls.append(1) or "more"))
        hops = agent_stream.MAX_TOOL_HOPS
        llm = ScriptedLLM(
            *[_calls(("again", {})) for _ in range(hops)],
            _text("Here's what I found."),
        )

        result = self.run_turn(llm)

        self.assertEqual(result, "Here's what I found.")
        self.assertEqual(len(calls), hops)
        self.assertEqual(len(llm.requests), hops + 1)
        self.assertIsNone(llm.requests[-1]["tools"])

    def test_tool_calls_on_the_final_hop_are_ignored(self):
        """Even if a model emits a call with no tools offered, nothing executes."""
        calls = []
        registry.register(_tool("again", lambda a, c: calls.append(1) or "more"))
        hops = agent_stream.MAX_TOOL_HOPS
        stubborn = [{"message": {"content": "Best I can do.", "tool_calls": [
            {"function": {"name": "again", "arguments": {}}}
        ]}}]
        llm = ScriptedLLM(*[_calls(("again", {})) for _ in range(hops)], stubborn)

        self.assertEqual(self.run_turn(llm), "Best I can do.")
        self.assertEqual(len(calls), hops)


class MalformedCallTests(AgentLoopTestCase):
    def test_malformed_arguments_get_one_corrective_retry(self):
        registry.register(_tool("t", lambda a, c: "ok"))
        bad = [{"message": {"content": "", "tool_calls": [
            {"function": {"name": "t", "arguments": "not-an-object"}}
        ]}}]
        llm = ScriptedLLM(bad, _calls(("t", {})), _text("Fixed it."))

        self.assertEqual(self.run_turn(llm), "Fixed it.")
        self.assertIn("invalid", llm.requests[1]["messages"][-1]["content"])

    def test_gives_up_after_second_malformed_call(self):
        bad = [{"message": {"content": "", "tool_calls": [
            {"function": {"name": "t", "arguments": "nope"}}
        ]}}]
        llm = ScriptedLLM(bad, bad)

        self.assertEqual(self.run_turn(llm), "I couldn't figure out how to do that.")


if __name__ == "__main__":
    unittest.main()
