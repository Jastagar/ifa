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
    return np.full(CAPTURE_SAMPLES, value, dtype=np.float32)


def _reader_yielding(chunks: list[np.ndarray]):
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
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        probs = [0.9, 0.9, 0.9] + [0.01] * silence_chunks_needed
        chunks = [_chunk(v) for v in [0.5, 0.5, 0.5] + [0.0] * silence_chunks_needed]

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            vad=_FakeVAD(probs),
        )

        self.assertEqual(audio.dtype, np.float32)
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
        probs = [0.9] * 10_000
        chunks = [_chunk(0.5)] * 10_000

        fake_time = [0.0]
        calls = [0]

        def fake_monotonic():
            calls[0] += 1
            fake_time[0] += 0.032
            return fake_time[0]

        with patch("ifa.voice.capture.time.monotonic", side_effect=fake_monotonic):
            audio = capture_utterance(
                read_chunk=_reader_yielding(chunks),
                max_utterance_ms=200,
                vad=_FakeVAD(probs),
            )
        self.assertLess(len(audio), 20 * CAPTURE_SAMPLES)
        self.assertGreaterEqual(len(audio), 3 * CAPTURE_SAMPLES)


class CaptureSilenceHandlingTests(unittest.TestCase):
    def test_hesitation_before_speech_does_not_cut_turn(self):
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        probs = (
            [0.01] * 80
            + [0.9] * 3
            + [0.01] * silence_chunks_needed
        )
        chunks = [_chunk(0.0)] * len(probs)

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            max_utterance_ms=60_000,
            vad=_FakeVAD(probs),
        )
        self.assertEqual(
            len(audio),
            (80 + 3 + silence_chunks_needed) * CAPTURE_SAMPLES,
        )

    def test_speech_resets_silence_counter_mid_utterance(self):
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        short_pause_chunks = 20
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
        self.assertEqual(
            len(audio),
            (3 + short_pause_chunks + 3 + silence_chunks_needed) * CAPTURE_SAMPLES,
        )


class CaptureEnvOverrideTests(unittest.TestCase):
    def test_silence_ms_env_override(self):
        short_silence_chunks = (200 // CHUNK_MS) + 1
        probs = [0.9] + [0.01] * short_silence_chunks
        chunks = [_chunk(0.0)] * len(probs)

        with patch.dict(os.environ, {"IFA_VAD_SILENCE_MS": "200"}):
            audio = capture_utterance(
                read_chunk=_reader_yielding(chunks),
                vad=_FakeVAD(probs),
            )
        self.assertLessEqual(
            len(audio), (1 + short_silence_chunks + 2) * CAPTURE_SAMPLES
        )

    def test_threshold_env_override(self):
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        probs = [0.3] * 5 + [0.0] * silence_chunks_needed
        chunks = [_chunk(0.0)] * len(probs)

        with patch.dict(os.environ, {"IFA_VAD_THRESHOLD": "0.2"}):
            audio = capture_utterance(
                read_chunk=_reader_yielding(chunks),
                vad=_FakeVAD(probs),
            )
        self.assertEqual(
            len(audio), (5 + silence_chunks_needed) * CAPTURE_SAMPLES
        )


class CaptureStartTimeoutTests(unittest.TestCase):
    def test_start_timeout_returns_empty_on_extended_silence(self):
        probs = [0.01] * 10_000
        chunks = [_chunk(0.0)] * 10_000

        fake_time = [0.0]
        def fake_monotonic():
            fake_time[0] += 0.032
            return fake_time[0]

        with patch("ifa.voice.capture.time.monotonic", side_effect=fake_monotonic):
            audio = capture_utterance(
                read_chunk=_reader_yielding(chunks),
                start_timeout_ms=500,
                vad=_FakeVAD(probs),
            )
        self.assertEqual(len(audio), 0)

    def test_start_timeout_does_not_cut_once_speech_started(self):
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        probs = [0.9] * 3 + [0.01] * silence_chunks_needed
        chunks = [_chunk(0.5)] * len(probs)

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            start_timeout_ms=100,
            vad=_FakeVAD(probs),
        )
        self.assertEqual(
            len(audio), (3 + silence_chunks_needed) * CAPTURE_SAMPLES
        )


class CaptureFallbackTests(unittest.TestCase):
    def test_vad_exception_falls_back_to_energy_threshold(self):
        silence_chunks_needed = (1500 // CHUNK_MS) + 1
        loud_chunk = np.full(CAPTURE_SAMPLES, 0.3, dtype=np.float32)
        quiet_chunk = np.zeros(CAPTURE_SAMPLES, dtype=np.float32)
        chunks = [loud_chunk] * 3 + [quiet_chunk] * silence_chunks_needed

        vad = _FakeVAD([])
        vad.raise_next = True

        audio = capture_utterance(
            read_chunk=_reader_yielding(chunks),
            vad=vad,
        )
        self.assertEqual(
            len(audio), (3 + silence_chunks_needed) * CAPTURE_SAMPLES
        )


if __name__ == "__main__":
    unittest.main()
