"""Tests for ifa.voice.capture.capture_utterance.

The real Silero VAD model is replaced with a callable mock that returns
a scripted sequence of probabilities. This keeps tests fast and
deterministic.

Run: PYTHONPATH=. python -m unittest ifa.tests.test_capture -v
"""
import os
import unittest
from unittest.mock import patch

import numpy as np

from ifa.voice.capture import (
    CAPTURE_SAMPLES,
    CHUNK_MS,
    capture_utterance,
)


class _FakeVAD:
    """Returns scripted per-chunk probabilities, mimicking SileroVADModel.__call__."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self._idx = 0
        self.raise_next = False

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        if self.raise_next:
            raise RuntimeError("simulated VAD failure")
        if self._idx >= len(self._probs):
            return np.array([0.0], dtype=np.float32)
        p = self._probs[self._idx]
        self._idx += 1
        return np.array([p], dtype=np.float32)


def _chunk(value: float = 0.0) -> np.ndarray:
    """Return a 512-sample float32 chunk filled with ``value`` (so tests can
    visually inspect what got buffered)."""
    return np.full(CAPTURE_SAMPLES, value, dtype=np.float32)


def _reader_yielding(chunks: list[np.ndarray]):
    """Wrap a list of chunks into a read_chunk() callable. After the list
    is exhausted, yields zero-filled chunks so the capture loop can't hang."""
    idx = [0]
    def read():
        if idx[0] < len(chunks):
            c = chunks[idx[0]]
            idx[0] += 1
            return c
        return _chunk(0.0)
    return read


class CaptureHappyPathTests(unittest.TestCase):
    def test_returns_after_configured_silence_window(self):
        """3 speech chunks, then enough silence chunks to exceed 1500ms."""
        silence_chunks_needed = (1500 // CHUNK_MS) + 1  # ~47 chunks at 32ms
        probs = [0.9, 0.9, 0.9] + [0.01] * silence_chunks_needed
        # One chunk per prob
        chunks = [_chunk(v) for v in [0.5, 0.5, 0.5] + [0.0] * silence_chunks_needed]

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            vad=_FakeVAD(probs),
        )

        self.assertEqual(audio.dtype, np.float32)
        # Should include all 3 speech chunks + all silence chunks up to break
        self.assertEqual(
            len(audio), (3 + silence_chunks_needed) * CAPTURE_SAMPLES
        )

    def test_int16_input_coerced_to_float32(self):
        probs = [0.9] + [0.01] * 60
        int16_chunk = np.zeros(CAPTURE_SAMPLES, dtype=np.int16)
        chunks = [int16_chunk] + [int16_chunk] * 60

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            vad=_FakeVAD(probs),
        )
        self.assertEqual(audio.dtype, np.float32)

    def test_max_utterance_cap_truncates(self):
        """If the user never goes silent, capture returns at max cap."""
        probs = [0.9] * 10_000  # always speaking
        chunks = [_chunk(0.5)] * 10_000

        # Fake time: each read advances the clock by the expected chunk duration.
        fake_time = [0.0]
        calls = [0]

        def fake_monotonic():
            calls[0] += 1
            fake_time[0] += 0.032  # 32ms per chunk
            return fake_time[0]

        with patch("ifa.voice.capture.time.monotonic", side_effect=fake_monotonic):
            audio = capture_utterance(
                read_chunk=_reader_yielding(chunks),
                max_utterance_ms=200,  # ~6 chunks at 32ms
                vad=_FakeVAD(probs),
            )
        # Cap at 200ms; each iteration is 32ms + some monotonic overhead.
        # Expect capture to end within ~10 chunks.
        self.assertLess(len(audio), 20 * CAPTURE_SAMPLES)
        self.assertGreaterEqual(len(audio), 3 * CAPTURE_SAMPLES)


class CaptureSilenceHandlingTests(unittest.TestCase):
    def test_hesitation_before_speech_does_not_cut_turn(self):
        """Silence before any speech must NOT trigger the silence timer.

        Prevents the bug where a user pauses to think before replying
        and the capture cuts off before they speak.
        """
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        # 80 initial silence chunks (~2.5s of hesitation) then speech then trail-silence
        probs = (
            [0.01] * 80  # hesitation
            + [0.9] * 3  # speech
            + [0.01] * silence_chunks_needed  # trailing silence
        )
        chunks = [_chunk(0.0)] * len(probs)

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            max_utterance_ms=60_000,
            vad=_FakeVAD(probs),
        )
        # The 80 hesitation chunks are included in the output (we keep
        # everything from wake onward), but the turn did NOT cut during them.
        self.assertEqual(
            len(audio),
            (80 + 3 + silence_chunks_needed) * CAPTURE_SAMPLES,
        )

    def test_speech_resets_silence_counter_mid_utterance(self):
        """Brief pause mid-speech doesn't cut the turn prematurely."""
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        # pattern: speech, short pause (< 1.5s), speech, then proper silence
        short_pause_chunks = 20  # ~640ms, well under 1.5s
        probs = (
            [0.9] * 3
            + [0.01] * short_pause_chunks
            + [0.9] * 3
            + [0.01] * silence_chunks_needed
        )
        chunks = [_chunk(0.0)] * len(probs)

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            vad=_FakeVAD(probs),
        )
        # Entire utterance should be captured — the mid-pause didn't cut it
        self.assertEqual(
            len(audio),
            (3 + short_pause_chunks + 3 + silence_chunks_needed) * CAPTURE_SAMPLES,
        )


class CaptureEnvOverrideTests(unittest.TestCase):
    def test_silence_ms_env_override(self):
        """IFA_VAD_SILENCE_MS shortens or lengthens the silence window."""
        # With 200ms silence requirement, only ~7 silence chunks are needed
        short_silence_chunks = (200 // CHUNK_MS) + 1
        probs = [0.9] + [0.01] * short_silence_chunks
        chunks = [_chunk(0.0)] * len(probs)

        with patch.dict(os.environ, {"IFA_VAD_SILENCE_MS": "200"}):
            audio = capture_utterance(
                read_chunk=_reader_yielding(chunks),
                vad=_FakeVAD(probs),
            )
        # Should cut after the ~7 silence chunks, not 47
        self.assertLessEqual(
            len(audio), (1 + short_silence_chunks + 2) * CAPTURE_SAMPLES
        )

    def test_threshold_env_override(self):
        """IFA_VAD_THRESHOLD changes what counts as speech."""
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        # With default threshold 0.5, a prob of 0.3 is silence; with
        # threshold 0.2 it becomes speech.
        probs = [0.3] * 5 + [0.0] * silence_chunks_needed
        chunks = [_chunk(0.0)] * len(probs)

        with patch.dict(os.environ, {"IFA_VAD_THRESHOLD": "0.2"}):
            audio = capture_utterance(
                read_chunk=_reader_yielding(chunks),
                vad=_FakeVAD(probs),
            )
        # With lowered threshold the first 5 chunks register as speech,
        # then trailing silence cuts the turn.
        self.assertEqual(
            len(audio), (5 + silence_chunks_needed) * CAPTURE_SAMPLES
        )


class CaptureFallbackTests(unittest.TestCase):
    def test_vad_exception_falls_back_to_energy_threshold(self):
        """If VAD raises, capture completes via energy fallback without erroring."""
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        # Speech chunks have noticeable amplitude, silence chunks are zero-valued.
        loud_chunk = np.full(CAPTURE_SAMPLES, 0.3, dtype=np.float32)
        quiet_chunk = np.zeros(CAPTURE_SAMPLES, dtype=np.float32)
        chunks = [loud_chunk] * 3 + [quiet_chunk] * silence_chunks_needed

        vad = _FakeVAD([])
        vad.raise_next = True  # every call raises

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            vad=vad,
        )
        # The energy-threshold fallback sees loud (~RMS 0.3) as speech and
        # quiet (RMS 0) as silence, so the turn captures and cuts correctly.
        self.assertEqual(
            len(audio), (3 + silence_chunks_needed) * CAPTURE_SAMPLES
        )


if __name__ == "__main__":
    unittest.main()
