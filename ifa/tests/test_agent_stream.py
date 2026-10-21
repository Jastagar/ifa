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
    def __init__(self, *replies):
        self._replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, model, messages, tools=None, **_):
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
        for target, value in [
            ("ifa.core.agent_stream.VibeManager.get_system_based_on_vibe", ""),
            ("ifa.core.agent_stream.memory_store.prompt_context", ""),
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
        self.assertIn("ERROR 404", _tool_messages(llm.requests[1])[0]["content"])
        self.assertTrue(all(r["tools"] for r in llm.requests))

    def test_tool_calls_are_announced_before_they_run(self):
        events = []
        registry.register(_tool("get_time", lambda a, c: events.append("ran") or "3 PM"))
        llm = ScriptedLLM(_calls(("get_time", {})), _text("It's 3 PM."))

        with patch("ifa.core.agent_stream.stream_chat", side_effect=llm):
            agent_stream.agent_turn_stream(
                "time?", self.ctx, Memory(),
                on_tool_calls=lambda names: events.append(("announced", names)),
            )

        self.assertEqual(events, [("announced", ["get_time"]), "ran"])

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


class TerminalToolTests(AgentLoopTestCase):
    def _terminal(self, name, handler):
        tool = _tool(name, handler)
        tool.return_direct = True
        return tool

    def test_terminal_tool_result_is_spoken_without_another_llm_call(self):
        registry.register(self._terminal("read_notifications", lambda a, c: "Joe wants to know if you're free tonight."))
        spoken = []
        llm = ScriptedLLM(_calls(("read_notifications", {})))

        result = self.run_turn(llm, "What is it?", spoken=spoken)

        self.assertEqual(result, "Joe wants to know if you're free tonight.")
        self.assertEqual(spoken, ["Joe wants to know if you're free tonight."])
        self.assertEqual(len(llm.requests), 1)

    def test_terminal_tool_error_goes_back_to_the_model(self):
        registry.register(self._terminal("read_notifications", lambda a, c: 1 / 0))
        llm = ScriptedLLM(_calls(("read_notifications", {})), _text("I couldn't read them."))

        self.assertEqual(self.run_turn(llm), "I couldn't read them.")
        self.assertEqual(len(llm.requests), 2)

    def test_mixed_with_a_normal_tool_uses_the_normal_flow(self):
        registry.register(self._terminal("read_notifications", lambda a, c: "summary"))
        registry.register(_tool("get_time", lambda a, c: "3 PM"))
        llm = ScriptedLLM(_calls(("read_notifications", {}), ("get_time", {})), _text("Both done."))

        self.assertEqual(self.run_turn(llm), "Both done.")


class FabricatedToolResultTests(AgentLoopTestCase):
    FAKE = [
        {"message": {"content": "I'll check. "}},
        {"message": {"content": "<TOOL"}},
        {"message": {"content": "_RESULT_abc_START tool=read_notifications>Report due at 5 PM. "}},
        {"message": {"content": "Joe says hi.<TOOL_RESULT_abc_END>"}},
    ]

    def test_fake_result_is_not_spoken_and_model_is_asked_for_a_real_call(self):
        registry.register(_tool("read_notifications", lambda a, c: "Joe wants to know if you're free."))
        spoken = []
        llm = ScriptedLLM(self.FAKE, _calls(("read_notifications", {})), _text("Joe wants to know if you're free."))

        result = self.run_turn(llm, "What is it?", spoken=spoken)

        self.assertEqual(result, "Joe wants to know if you're free.")
        self.assertEqual(spoken, ["Joe wants to know if you're free."])
        retry_request = llm.requests[1]["messages"]
        self.assertIn("you wrote a tool result yourself", retry_request[-1]["content"])
        self.assertFalse(any("Report due" in (m.get("content") or "") for m in retry_request))

    def test_gives_up_after_a_second_fabrication(self):
        llm = ScriptedLLM(self.FAKE, self.FAKE)
        self.assertEqual(self.run_turn(llm), "I couldn't figure out how to do that.")


class BudgetTests(AgentLoopTestCase):
    def test_final_hop_withholds_tools_so_model_must_answer(self):
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


class ActionClaimTests(AgentLoopTestCase):
    def setUp(self):
        super().setUp()
        self.played = []
        registry.register(_tool("spotify_control", lambda args, ctx: self.played.append(args) or "Playing Kesariya by Arijit Singh."))

    def run_with(self, llm, text, memory, spoken):
        with patch("ifa.core.agent_stream.stream_chat", side_effect=llm):
            return agent_stream.agent_turn_stream(text, self.ctx, memory, on_sentence=spoken.append)

    def test_claiming_an_action_without_a_tool_is_silenced_and_retried(self):
        spoken = []
        llm = ScriptedLLM(
            _text("Playing Kesariya by Arijit Singh, Sir."),
            _calls(("spotify_control", {"action": "play", "query": "kesariya"})),
            _text("Kesariya's playing, Sir."),
        )
        reply = self.run_with(llm, "play kesariya", Memory(), spoken)
        self.assertEqual(spoken, ["Kesariya's playing, Sir."])
        self.assertEqual(reply, "Kesariya's playing, Sir.")
        self.assertEqual(len(self.played), 1)
        self.assertIn("no tool was called", llm.requests[1]["messages"][-1]["content"])

    def test_after_the_tool_ran_the_confirmation_is_not_checked(self):
        spoken = []
        llm = ScriptedLLM(
            _calls(("spotify_control", {"action": "play", "query": "kesariya"})),
            _text("Playing Kesariya, Sir."),
        )
        self.run_with(llm, "play kesariya", Memory(), spoken)
        self.assertEqual(spoken, ["Playing Kesariya, Sir."])
        self.assertEqual(len(llm.requests), 2)

    def test_chat_that_merely_looks_like_a_claim_is_spoken_after_one_retry(self):
        spoken = []
        llm = ScriptedLLM(_text("Running is great for your heart."), _text("Running is great for your heart."))
        self.run_with(llm, "is running healthy?", Memory(), spoken)
        self.assertEqual(spoken, ["Running is great for your heart."])

    def test_ordinary_answers_pass_untouched(self):
        spoken = []
        llm = ScriptedLLM(_text("All good, Sir. How about you?"))
        self.run_with(llm, "kya haal hai?", Memory(), spoken)
        self.assertEqual(spoken, ["All good, Sir.", "How about you?"])
        self.assertEqual(len(llm.requests), 1)


class MemoryTests(AgentLoopTestCase):
    def test_history_keeps_the_tool_calls_so_the_next_turn_sees_them(self):
        registry.register(_tool("spotify_control", lambda args, ctx: "Playing Tum Hi Ho by Arijit Singh.\nextra detail"))
        memory = Memory()
        first = ScriptedLLM(_calls(("spotify_control", {"action": "play", "choice": 1})), _text("Playing it, Sir."))
        with patch("ifa.core.agent_stream.stream_chat", side_effect=first):
            agent_stream.agent_turn_stream("play tum hi ho", self.ctx, memory)
        second = ScriptedLLM(_text("Hello."))
        with patch("ifa.core.agent_stream.stream_chat", side_effect=second):
            agent_stream.agent_turn_stream("hi", self.ctx, memory)
        history = second.requests[0]["messages"][1:-1]
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(history[1]["tool_calls"][0]["function"]["name"], "spotify_control")
        self.assertEqual(history[2]["content"], "Playing Tum Hi Ho by Arijit Singh.")
        self.assertNotIn(agent_stream.TOOL_RESULT_MARKER, str(history))

    def test_keeps_whole_recent_turns(self):
        memory = Memory(max_turns=2)
        for i in range(3):
            memory.add_turn(f"q{i}", [], f"a{i}")
        self.assertEqual([m["content"] for m in memory.recent()], ["q1", "a1", "q2", "a2"])


class UnaskedNotificationTests(AgentLoopTestCase):
    def test_reading_notifications_unasked_is_refused_and_the_real_question_answered(self):
        from ifa.services.notifications import NotificationCenter, NotificationInbox
        from ifa.tools import notifications as notifications_tool

        inbox = NotificationInbox(max_recent=7, ttl_sec=90, settle_sec=4, max_discarded=50)
        inbox.add(app="WhatsApp", package="com.whatsapp", sender="Joe", text="free tonight?", priority="normal")
        self.ctx.notifications = NotificationCenter(inbox, classify=lambda b: False)
        registry.register(notifications_tool.TOOL)
        spoken = []
        llm = ScriptedLLM(_calls(("read_notifications", {"scope": "new"})), _text("It's sunny and 31 degrees, Sir."))
        reply = self.run_turn(llm, "what's the weather like?", spoken)
        self.assertEqual(reply, "It's sunny and 31 degrees, Sir.")
        self.assertEqual(spoken, ["It's sunny and 31 degrees, Sir."])
        self.assertIn("didn't ask about notifications", _tool_messages(llm.requests[1])[0]["content"])
        self.assertEqual(inbox.unread_count(), 1)
        self.assertNotIn("unread phone notification", llm.requests[0]["messages"][-1]["content"])


class ActionClaimPatternTests(unittest.TestCase):
    def test_claims_and_promises_are_caught(self):
        for text in ("Playing Kesariya, Sir.", "Discord's open, Sir.", "Let me check your new notifications for you.",
                     "I'll send that message to John for you.", "Got it, Sir. I'll remember that.",
                     "Okay, I'm going to open YouTube."):
            self.assertTrue(agent_stream.ACTION_CLAIM.search(text), text)

    def test_ordinary_answers_are_not(self):
        for text in ("All good, Sir. How about you?", "It's 4:35 PM, Sir.", "A black hole is a region of space.",
                     "I'll need the time for that reminder, Sir.", "Who should I send it to?"):
            self.assertFalse(agent_stream.ACTION_CLAIM.search(text), text)
