"""Tests for filler clips: which phrase is chosen, and when run_turn plays one.

Run: python -m unittest ifa.tests.test_fillers -v
"""
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from ifa.core import orchestrator
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.services.fillers import Fillers


def _ready_fillers(generic=("One moment.", "Let me see."), by_tool=None) -> Fillers:
    fillers = Fillers(
        synthesize=lambda text: f"<audio:{text}>",
        generic=generic,
        by_tool=by_tool or {"get_time": "Checking the time."},
        rng=random.Random(0),
    )
    fillers.prepare_in_background().join()
    return fillers


class ChooseTests(unittest.TestCase):
    def test_tool_phrase_wins_when_a_tool_matches(self):
        phrase, audio = _ready_fillers().choose(["get_time"])
        self.assertEqual(phrase, "Checking the time.")
        self.assertEqual(audio, "<audio:Checking the time.>")

    def test_unknown_tool_falls_back_to_a_generic_phrase(self):
        phrase, _ = _ready_fillers().choose(["some_new_tool"])
        self.assertIn(phrase, ("One moment.", "Let me see."))

    def test_never_repeats_the_same_generic_phrase_twice_in_a_row(self):
        fillers = _ready_fillers()
        picks = [fillers.choose()[0] for _ in range(20)]
        self.assertTrue(all(a != b for a, b in zip(picks, picks[1:])))

    def test_nothing_rendered_yet_means_no_filler(self):
        fillers = Fillers(synthesize=lambda text: "audio")  # never prepared
        self.assertIsNone(fillers.choose())
        self.assertIsNone(fillers.choose(["get_time"]))

    def test_a_phrase_that_fails_to_render_is_skipped(self):
        def synthesize(text):
            if text == "One moment.":
                raise RuntimeError("TTS hiccup")
            return "audio"

        fillers = Fillers(synthesize, generic=["One moment.", "Let me see."], by_tool={})
        fillers.prepare_in_background().join()
        self.assertEqual(fillers.choose()[0], "Let me see.")


class RunTurnFillerTests(unittest.TestCase):
    """run_turn with a scripted agent: when does a filler play, and in what order?"""

    def setUp(self):
        self.spoken: list[str] = []  # everything that reached TTS, in order
        self.tts = MagicMock()
        self.tts.play_clip.side_effect = lambda audio, label: self.spoken.append(f"filler:{label}")
        self.ctx = AgentContext(tts=self.tts, db_path=":memory:")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.executor.shutdown)

    def run_turn(self, agent, filler_delay=0.05, fillers=None):
        with patch.object(orchestrator, "agent_turn_stream", agent):
            return orchestrator.run_turn(
                "hi", self.ctx, Memory(), self.spoken.append,
                fillers or _ready_fillers(), self.executor, filler_delay=filler_delay,
            )

    def test_fast_answer_gets_no_filler(self):
        def agent(text, ctx, memory, on_sentence, on_tool_calls):
            on_sentence("Hello.")

        self.run_turn(agent, filler_delay=1.0)
        self.assertEqual(self.spoken, ["Hello."])

    def test_slow_answer_gets_one_generic_filler_before_it(self):
        def agent(text, ctx, memory, on_sentence, on_tool_calls):
            time.sleep(0.2)
            on_sentence("Here you go.")

        timer = self.run_turn(agent)
        self.assertEqual(len(self.spoken), 2)
        self.assertTrue(self.spoken[0].startswith("filler:"))
        self.assertEqual(self.spoken[1], "Here you go.")
        self.assertEqual(timer.filler_text, self.spoken[0].removeprefix("filler:"))

    def test_tool_call_plays_the_tool_filler_immediately_and_only_once(self):
        played_before_delay = threading.Event()

        def agent(text, ctx, memory, on_sentence, on_tool_calls):
            on_tool_calls(["get_time"])
            if self.spoken:
                played_before_delay.set()
            time.sleep(0.2)  # well past filler_delay: the generic one must not also play
            on_sentence("It's 3 PM.")

        self.run_turn(agent)
        self.assertTrue(played_before_delay.is_set())
        self.assertEqual(self.spoken, ["filler:Checking the time.", "It's 3 PM."])

    def test_no_filler_once_the_answer_has_started(self):
        def agent(text, ctx, memory, on_sentence, on_tool_calls):
            on_sentence("Let me check.")  # model spoke a preamble first
            on_tool_calls(["get_time"])
            on_sentence("It's 3 PM.")

        self.run_turn(agent, filler_delay=1.0)
        self.assertEqual(self.spoken, ["Let me check.", "It's 3 PM."])

    def test_filler_not_ready_means_silence_not_a_crash(self):
        def agent(text, ctx, memory, on_sentence, on_tool_calls):
            time.sleep(0.2)
            on_sentence("Done.")

        self.run_turn(agent, fillers=Fillers(synthesize=lambda t: "audio"))
        self.assertEqual(self.spoken, ["Done."])
        self.tts.play_clip.assert_not_called()


if __name__ == "__main__":
    unittest.main()
