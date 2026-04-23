"""Tests for ifa.voice.wake_word.WakeWordListener.

openWakeWord is mocked so tests run without the model files and
without network access.

Run: PYTHONPATH=. python -m unittest ifa.tests.test_wake_word -v
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _install_fake_openwakeword() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Install a fake openwakeword package in sys.modules.

    Returns (download_models mock, Model class mock, Model instance mock).
    Call sys.modules cleanup via the returned teardown function.
    """
    fake_pkg = types.ModuleType("openwakeword")
    fake_model_mod = types.ModuleType("openwakeword.model")
    fake_utils_mod = types.ModuleType("openwakeword.utils")

    download_mock = MagicMock(name="download_models")
    model_instance = MagicMock(name="Model_instance")
    model_instance.predict = MagicMock(return_value={"hey_mycroft": 0.0})
    model_class = MagicMock(name="Model_class", return_value=model_instance)

    fake_utils_mod.download_models = download_mock
    fake_model_mod.Model = model_class

    sys.modules["openwakeword"] = fake_pkg
    sys.modules["openwakeword.model"] = fake_model_mod
    sys.modules["openwakeword.utils"] = fake_utils_mod

    return download_mock, model_class, model_instance


def _uninstall_fake_openwakeword() -> None:
    for name in ("openwakeword", "openwakeword.model", "openwakeword.utils"):
        sys.modules.pop(name, None)


class _FakeTTS:
    """Minimal TTSService stand-in with a mutable is_speaking flag."""

    def __init__(self, speaking: bool = False) -> None:
        self._speaking = speaking

    @property
    def is_speaking(self) -> bool:
        return self._speaking


class WakeWordListenerInitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.download, self.model_cls, self.model_inst = _install_fake_openwakeword()
        # Make predict return the right key for the default ("hey_mycroft") model
        self.model_inst.predict.return_value = {"hey_mycroft": 0.0}

    def tearDown(self) -> None:
        _uninstall_fake_openwakeword()

    def test_init_downloads_and_instantiates_default_model_onnx(self):
        from ifa.voice.wake_word import WakeWordListener

        # Default is "hey_mycroft"; clear any env override that might be set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IFA_WAKE_MODEL", None)
            listener = WakeWordListener(tts_service=None)

        self.assertEqual(listener.score_key, "hey_mycroft")
        self.download.assert_called_once_with(model_names=["hey_mycroft"])
        self.model_cls.assert_called_once()
        kwargs = self.model_cls.call_args.kwargs
        self.assertEqual(kwargs["wakeword_models"], ["hey_mycroft"])
        self.assertEqual(kwargs["inference_framework"], "onnx")

    def test_ifa_wake_model_env_override_selects_builtin(self):
        from ifa.voice.wake_word import WakeWordListener

        # Use a non-default to prove the override actually takes effect.
        with patch.dict(os.environ, {"IFA_WAKE_MODEL": "alexa"}):
            listener = WakeWordListener(tts_service=None)

        self.assertEqual(listener.score_key, "alexa")
        self.download.assert_called_once_with(model_names=["alexa"])
        self.assertEqual(
            self.model_cls.call_args.kwargs["wakeword_models"], ["alexa"]
        )

    def test_custom_onnx_path_skips_download_and_derives_score_key(self):
        """Future hey_ifa path: user points IFA_WAKE_MODEL at a .onnx file."""
        from ifa.voice.wake_word import WakeWordListener

        # Use a path that actually exists so os.path.exists returns True
        # (any repo file works; the fake Model doesn't actually load it)
        import pathlib
        existing = str(pathlib.Path(__file__).resolve())
        fake_custom = existing  # pretend this is our hey_ifa.onnx

        with patch.dict(os.environ, {"IFA_WAKE_MODEL": fake_custom}):
            listener = WakeWordListener(tts_service=None)

        self.download.assert_not_called()  # paths don't download
        self.assertEqual(
            self.model_cls.call_args.kwargs["wakeword_models"], [fake_custom]
        )
        # Score key is basename without extension
        expected_key = pathlib.Path(existing).stem
        self.assertEqual(listener.score_key, expected_key)

    def test_model_spec_kwarg_overrides_env(self):
        from ifa.voice.wake_word import WakeWordListener

        # Env says alexa, kwarg says hey_jarvis — kwarg wins.
        with patch.dict(os.environ, {"IFA_WAKE_MODEL": "alexa"}):
            listener = WakeWordListener(tts_service=None, model_spec="hey_jarvis")
        self.assertEqual(listener.score_key, "hey_jarvis")

    def test_init_reads_threshold_from_env(self):
        from ifa.voice.wake_word import WakeWordListener

        with patch.dict(os.environ, {"IFA_WAKE_THRESHOLD": "0.75"}):
            listener = WakeWordListener(tts_service=None)
        self.assertEqual(listener.threshold, 0.75)

    def test_init_threshold_kwarg_overrides_env(self):
        from ifa.voice.wake_word import WakeWordListener

        with patch.dict(os.environ, {"IFA_WAKE_THRESHOLD": "0.75"}):
            listener = WakeWordListener(tts_service=None, threshold=0.9)
        self.assertEqual(listener.threshold, 0.9)

    def test_init_raises_wakeword_init_error_on_download_failure(self):
        from ifa.voice.wake_word import WakeWordListener, WakeWordInitError

        self.download.side_effect = OSError("offline")
        with self.assertRaises(WakeWordInitError) as cm:
            WakeWordListener(tts_service=None)
        # Message should name the model + hint at first-run network requirement
        msg = str(cm.exception)
        self.assertIn("hey_mycroft", msg)
        self.assertIn("internet", msg.lower())


class WakeWordListenerDetectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.download, self.model_cls, self.model_inst = _install_fake_openwakeword()

    def tearDown(self) -> None:
        _uninstall_fake_openwakeword()

    def _make_read_chunk(self, n_silent: int):
        """Returns a callable that yields n_silent silent frames, then
        one 'triggering' frame, then raises if called further."""
        frames_yielded = [0]
        def read():
            if frames_yielded[0] > n_silent:
                raise AssertionError(
                    "wait_for_wake should have returned already; read called too many times"
                )
            chunk = np.zeros(1280, dtype=np.float32)
            frames_yielded[0] += 1
            return chunk
        return read, frames_yielded

    def test_wait_for_wake_returns_after_consecutive_frames_above_threshold(self):
        """Default requires 2 consecutive frames above threshold (IFA_WAKE_CONSECUTIVE)."""
        from ifa.voice.wake_word import WakeWordListener

        # Two low frames, then two high in a row = fires on the 4th
        self.model_inst.predict.side_effect = [
            {"hey_mycroft": 0.1},
            {"hey_mycroft": 0.4},
            {"hey_mycroft": 0.8},   # streak = 1, not yet enough
            {"hey_mycroft": 0.9},   # streak = 2, fires with this score
        ]

        # Ensure IFA_WAKE_CONSECUTIVE default is 2
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IFA_WAKE_CONSECUTIVE", None)
            listener = WakeWordListener(tts_service=None, threshold=0.5)

        read, frames_yielded = self._make_read_chunk(n_silent=10)
        score = listener.wait_for_wake(read_chunk=read)
        self.assertEqual(score, 0.9)
        self.assertEqual(frames_yielded[0], 4)
        self.assertEqual(self.model_inst.predict.call_count, 4)

    def test_single_frame_spike_does_not_fire_with_default_consecutive(self):
        """A one-frame spike above threshold surrounded by low scores must not fire."""
        from ifa.voice.wake_word import WakeWordListener

        # Pattern: high, low, high, high — only the final two-in-a-row fires
        self.model_inst.predict.side_effect = [
            {"hey_mycroft": 0.9},   # streak = 1
            {"hey_mycroft": 0.1},   # reset to 0
            {"hey_mycroft": 0.9},   # streak = 1
            {"hey_mycroft": 0.9},   # streak = 2, fires
        ]

        listener = WakeWordListener(tts_service=None, threshold=0.5)
        read, _ = self._make_read_chunk(n_silent=10)
        listener.wait_for_wake(read_chunk=read)
        self.assertEqual(self.model_inst.predict.call_count, 4)

    def test_consecutive_env_override_to_one_restores_single_frame_firing(self):
        from ifa.voice.wake_word import WakeWordListener

        self.model_inst.predict.side_effect = [
            {"hey_mycroft": 0.9},   # with CONSECUTIVE=1 this fires immediately
        ]
        with patch.dict(os.environ, {"IFA_WAKE_CONSECUTIVE": "1"}):
            listener = WakeWordListener(tts_service=None, threshold=0.5)
        read, _ = self._make_read_chunk(n_silent=10)
        listener.wait_for_wake(read_chunk=read)
        self.assertEqual(self.model_inst.predict.call_count, 1)

    def test_tts_speaking_resets_consecutive_streak(self):
        """If tts.is_speaking becomes True mid-streak, the streak must reset.

        predict IS called during mute (with silence, to keep the feature
        buffer flowing), but the returned score is ignored and the streak
        resets — so the first post-unmute score has to start from zero.
        """
        from ifa.voice.wake_word import WakeWordListener

        tts = _FakeTTS(speaking=False)
        listener = WakeWordListener(tts_service=tts, threshold=0.5)

        # Reads:
        #   1: live, 0.9 → streak=1
        #   2: muted → predict called with silence, score ignored, streak reset
        #   3: live, 0.9 → streak=1
        #   4: live, 0.9 → streak=2 → fires
        # So predict receives 4 calls total, 3 for live mic + 1 for silence.
        self.model_inst.predict.side_effect = [
            {"hey_mycroft": 0.9},   # live #1
            {"hey_mycroft": 0.0},   # silence call during mute (score ignored)
            {"hey_mycroft": 0.9},   # live #2
            {"hey_mycroft": 0.9},   # live #3 → fires
        ]

        call_counter = [0]
        def read():
            call_counter[0] += 1
            # On the 2nd read, TTS is speaking (mutes + resets streak)
            tts._speaking = call_counter[0] == 2
            return np.zeros(1280, dtype=np.float32)

        listener.wait_for_wake(read_chunk=read)
        self.assertEqual(call_counter[0], 4)
        # predict called once per read — 4 total (3 live + 1 silence)
        self.assertEqual(self.model_inst.predict.call_count, 4)

    def test_mic_chunks_replaced_with_silence_during_tts_speaking(self):
        """During mute, the listener feeds silence to the model (not the
        live mic chunk) and does not fire, but DOES keep the model's
        rolling feature buffer advancing.

        Earlier approach (skip predict entirely during mute) caused
        self-retriggering: stale wake-word audio remained in the buffer,
        and the first post-mute chunk would re-fire. This test locks in
        the fix.
        """
        from ifa.voice.wake_word import WakeWordListener

        tts = _FakeTTS(speaking=True)
        with patch.dict(os.environ, {"IFA_WAKE_CONSECUTIVE": "1"}):
            listener = WakeWordListener(tts_service=tts, threshold=0.5)

        self.model_inst.predict.return_value = {"hey_mycroft": 0.9}

        call_counter = [0]

        def read_then_unmute():
            if call_counter[0] == 3:
                tts._speaking = False
            call_counter[0] += 1
            # Return a distinctive pattern so we can see mic-vs-silence
            return np.full(1280, 0.5, dtype=np.float32)

        listener.wait_for_wake(read_chunk=read_then_unmute)

        # predict called for every read: 3 muted (with silence) + 1 unmuted = 4
        self.assertEqual(call_counter[0], 4)
        self.assertEqual(self.model_inst.predict.call_count, 4)

        # The first 3 calls must have received SILENCE (not the live mic)
        # even though is_speaking was True.
        muted_calls = self.model_inst.predict.call_args_list[:3]
        for call in muted_calls:
            arr = call[0][0]
            self.assertEqual(arr.dtype, np.int16)
            self.assertTrue(
                np.all(arr == 0),
                "muted predict must receive zeros, not the live mic chunk",
            )

        # The 4th (post-unmute) call received the actual live chunk (non-zero)
        live_call_arr = self.model_inst.predict.call_args_list[3][0][0]
        self.assertTrue(
            np.any(live_call_arr != 0),
            "post-unmute predict must receive the real mic chunk",
        )

    def test_model_reset_called_after_successful_detection(self):
        """Prevents self-retriggering on the next wait_for_wake cycle."""
        from ifa.voice.wake_word import WakeWordListener

        with patch.dict(os.environ, {"IFA_WAKE_CONSECUTIVE": "1"}):
            listener = WakeWordListener(tts_service=None, threshold=0.5)
        self.model_inst.predict.return_value = {"hey_mycroft": 0.9}

        listener.wait_for_wake(
            read_chunk=lambda: np.zeros(1280, dtype=np.float32)
        )
        self.model_inst.reset.assert_called_once()

    def test_int16_input_passes_through_without_reconversion(self):
        from ifa.voice.wake_word import WakeWordListener

        listener = WakeWordListener(tts_service=None, threshold=0.5)
        self.model_inst.predict.return_value = {"hey_mycroft": 0.9}

        int16_chunk = np.zeros(1280, dtype=np.int16)
        listener.wait_for_wake(read_chunk=lambda: int16_chunk)

        passed = self.model_inst.predict.call_args[0][0]
        self.assertEqual(passed.dtype, np.int16)

    def test_float32_input_converted_to_int16_pcm_range(self):
        from ifa.voice.wake_word import WakeWordListener

        listener = WakeWordListener(tts_service=None, threshold=0.5)
        self.model_inst.predict.return_value = {"hey_mycroft": 0.9}

        # Full-scale float32 sine-ish values in [-1, 1]
        float_chunk = np.linspace(-1.0, 1.0, 1280, dtype=np.float32)
        listener.wait_for_wake(read_chunk=lambda: float_chunk)

        passed = self.model_inst.predict.call_args[0][0]
        self.assertEqual(passed.dtype, np.int16)
        # -1.0 → -32768 (after clip); +1.0 → 32767 (after clip)
        self.assertEqual(passed.min(), -32768)
        self.assertLessEqual(passed.max(), 32767)
        self.assertGreaterEqual(passed.max(), 32700)

    def test_missing_score_key_treated_as_zero(self):
        """If predict() somehow returns a dict without the expected key,
        treat as below threshold rather than raising."""
        from ifa.voice.wake_word import WakeWordListener

        # CONSECUTIVE=1 isolates the missing-key behavior from streak logic.
        with patch.dict(os.environ, {"IFA_WAKE_CONSECUTIVE": "1"}):
            listener = WakeWordListener(tts_service=None, threshold=0.5)
        self.model_inst.predict.side_effect = [
            {},  # empty dict — should be treated as score 0.0
            {"hey_mycroft": 0.9},  # then a real detection
        ]
        call_counter = [0]

        def read():
            call_counter[0] += 1
            return np.zeros(1280, dtype=np.float32)

        listener.wait_for_wake(read_chunk=read)
        self.assertEqual(call_counter[0], 2)


if __name__ == "__main__":
    unittest.main()
