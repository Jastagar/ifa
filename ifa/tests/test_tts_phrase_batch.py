"""Regression tests for connected phrase-level TTS synthesis."""
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ifa.services.tts_service import TTSService


class PhraseBatchTests(unittest.TestCase):
    def test_phrases_are_split_for_prefetching(self):
        self.assertEqual(
            TTSService._split_sentences("First sentence. Second sentence."),
            ["First sentence.", "Second sentence."],
        )


if __name__ == "__main__":
    unittest.main()
