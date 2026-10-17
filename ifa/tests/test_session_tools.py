import os
import sys
import unittest
from unittest.mock import patch

from ifa.core.dismiss import is_dismissal
from ifa.supervisor import RESTART_EXIT_CODE, SUPERVISED_ENV, supervise


class DismissalTests(unittest.TestCase):
    def test_goodbyes_end_the_conversation(self):
        for text in ["That's all, thanks", "thanks", "Thank you so much, Ifa.", "ok thanks bye", "never mind",
                     "Nothing else for now", "bas", "theek hai bas ho gaya", "That's it for now, thank you",
                     "No thanks.", "okay that will be all", "Go to sleep Ifa", "That’s all"]:
            self.assertTrue(is_dismissal(text), text)

    def test_requests_and_short_answers_do_not(self):
        for text in ["no", "stop", "okay", "for now", "thanks, now play tum hi ho", "stop the music",
                     "that's all I wanted to ask, what time is it", "open youtube", "no don't run it",
                     "thanks for the song, skip it", ""]:
            self.assertFalse(is_dismissal(text), text)


class FakeChild:
    def __init__(self, code):
        self.code = code

    def wait(self):
        return self.code


class SupervisorTests(unittest.TestCase):
    def test_restarts_on_the_restart_code_and_stops_otherwise(self):
        codes = [RESTART_EXIT_CODE, RESTART_EXIT_CODE, 0]
        runs = []

        def run(argv, env):
            runs.append(env[SUPERVISED_ENV])
            return FakeChild(codes.pop(0))

        self.assertEqual(supervise(["python"], run=run), 0)
        self.assertEqual(runs, ["1", "1", "1"])

    def test_passes_through_a_crash_code(self):
        self.assertEqual(supervise(["python"], run=lambda argv, env: FakeChild(3)), 3)


class SessionToolTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from ifa.tools import session
        self.session = session
        self.ctx = SimpleNamespace(end_conversation=False, restart_requested=False)

    def test_end_conversation_is_silent(self):
        self.assertEqual(self.session.END_TOOL.handler({}, self.ctx), "")
        self.assertTrue(self.ctx.end_conversation)
        self.assertTrue(self.session.END_TOOL.return_direct)

    def test_restart_only_when_supervised(self):
        with patch.dict(os.environ, {SUPERVISED_ENV: ""}):
            self.assertTrue(self.session.RESTART_TOOL.handler({}, self.ctx).startswith("ERROR"))
            self.assertFalse(self.ctx.restart_requested)
        with patch.dict(os.environ, {SUPERVISED_ENV: "1"}):
            self.assertEqual(self.session.RESTART_TOOL.handler({}, self.ctx), "Restarting, Sir.")
            self.assertTrue(self.ctx.restart_requested)


if __name__ == "__main__":
    unittest.main()
