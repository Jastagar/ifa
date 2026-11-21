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

    def test_abbreviations_numbers_and_quotes_dont_split(self):
        cases = {
            "Dr. Rao is here. He said hi!": ["Dr. Rao is here.", "He said hi!"],
            "It costs 3.5 dollars. Okay?": ["It costs 3.5 dollars.", "Okay?"],
            "The U.S. Army won. Next.": ["The U.S. Army won.", "Next."],
            "Wait... What? Yes.": ["Wait...", "What?", "Yes."],
            'He said "Go now." Then left.': ['He said "Go now."', "Then left."],
            "Line one\nLine two.": ["Line one", "Line two."],
            "no punctuation at all": ["no punctuation at all"],
        }
        for text, expected in cases.items():
            self.assertEqual(TTSService._split_sentences(text), expected, text)


if __name__ == "__main__":
    unittest.main()


class StreamingSplitTests(unittest.TestCase):
    def test_streamed_text_never_breaks_inside_numbers_or_abbreviations(self):
        from ifa.utils.sentences import take_complete
        reply = "The driver is 572.60, out since Feb. 2025. Dr. Rao called at 3 p.m. today. Done!"
        buf, spoken = "", []
        for i in range(0, len(reply), 3):
            done, buf = take_complete(buf + reply[i:i + 3])
            spoken += done
        spoken.append(buf.strip())
        self.assertEqual(spoken, ["The driver is 572.60, out since Feb. 2025.",
                                  "Dr. Rao called at 3 p.m. today.", "Done!"])
