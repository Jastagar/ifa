"""Regression tests for the low-latency streaming agent path."""
import unittest
from unittest.mock import MagicMock, patch

from ifa.core import agent_stream
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.tools import registry


class DirectResponseTests(unittest.TestCase):
    def setUp(self):
        registry.clear()

    def tearDown(self):
        registry.clear()

    def test_direct_response_submits_each_sentence_without_waiting(self):
        """Every available sentence reaches the independent TTS pipeline."""
        ctx = AgentContext(tts=MagicMock(), db_path=":memory:", n8n_config={})
        spoken = []
        chunks = iter([
            {"message": {"content": "Hello there. "}},
            {"message": {"content": "How are you?"}},
        ])
        with patch("ifa.core.agent_stream.stream_chat", return_value=chunks) as stream:
            result = agent_stream.agent_turn_stream(
                "hi", ctx, Memory(), on_sentence=spoken.append
            )

        self.assertEqual(result, "Hello there. How are you?")
        self.assertEqual(spoken, ["Hello there.", "How are you?"])
        stream.assert_called_once()
        self.assertIsNotNone(stream.call_args.kwargs["tools"])


if __name__ == "__main__":
    unittest.main()
