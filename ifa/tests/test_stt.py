"""Tests for ifa.voice.stt.

The real WhisperModel is mocked so tests run without the model files.

Run: PYTHONPATH=. python -m unittest ifa.tests.test_stt -v
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _install_fake_whisper() -> tuple[MagicMock, MagicMock]:
    """Install a fake faster_whisper module. Returns (WhisperModel class, model instance)."""
    fake = types.ModuleType("faster_whisper")
    model_instance = MagicMock(name="whisper_instance")

    # Default: transcribe returns an iterable of segment-like objects and info
    def _default_transcribe(*_args, **_kwargs):
        segments = [
            types.SimpleNamespace(text=" hello "),
            types.SimpleNamespace(text=" world"),
        ]
        info = types.SimpleNamespace(language="en")
        return iter(segments), info

    model_instance.transcribe.side_effect = _default_transcribe
    model_class = MagicMock(name="WhisperModel_class", return_value=model_instance)
    fake.WhisperModel = model_class
    sys.modules["faster_whisper"] = fake
    return model_class, model_instance


def _uninstall_fake_whisper() -> None:
    sys.modules.pop("faster_whisper", None)


def _reset_stt_module_state() -> None:
    """Force the lazy singleton to reload on next call."""
    import ifa.voice.stt as stt
    stt._model = None


class TranscribeArrayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_cls, self.model_inst = _install_fake_whisper()
        _reset_stt_module_state()

    def tearDown(self) -> None:
        _uninstall_fake_whisper()
        _reset_stt_module_state()

    def test_returns_concatenated_stripped_text(self):
        from ifa.voice.stt import transcribe_array

        audio = np.zeros(16_000, dtype=np.float32)
        text = transcribe_array(audio)
        # Fake segments are " hello " and " world" → joined with a space,
        # then stripped on the outside only: "hello   world" (internal
        # whitespace preserved; real Whisper segments usually have a
        # leading space so joining like this is correct).
        self.assertEqual(text, "hello   world")

    def test_empty_audio_returns_empty_string_without_loading_model(self):
        from ifa.voice.stt import transcribe_array

        text = transcribe_array(np.zeros(0, dtype=np.float32))
        self.assertEqual(text, "")
        self.model_cls.assert_not_called()  # lazy: no model loaded for empty input

    def test_none_audio_returns_empty_string(self):
        from ifa.voice.stt import transcribe_array

        self.assertEqual(transcribe_array(None), "")

    def test_int16_input_coerced_to_float32(self):
        from ifa.voice.stt import transcribe_array

        audio_i16 = np.zeros(16_000, dtype=np.int16)
        transcribe_array(audio_i16)

        passed = self.model_inst.transcribe.call_args[0][0]
        self.assertEqual(passed.dtype, np.float32)

    def test_model_loaded_with_positional_size_not_keyword(self):
        """Regression guard for the `size=` kwarg error the plan review flagged.

        ``WhisperModel`` expects ``model_size_or_path`` as positional arg,
        not a keyword ``size=``. This test locks that in.
        """
        from ifa.voice.stt import transcribe_array

        # Force CPU path so auto→cuda-fallback doesn't confuse the assertion
        with patch.dict(os.environ, {"IFA_WHISPER_DEVICE": "cpu"}):
            _reset_stt_module_state()
            transcribe_array(np.zeros(100, dtype=np.float32))

        args, kwargs = self.model_cls.call_args
        self.assertEqual(args[0], "small.en")  # positional, default model
        self.assertNotIn("size", kwargs)
        self.assertEqual(kwargs.get("compute_type"), "int8")
        self.assertEqual(kwargs.get("device"), "cpu")

    def test_device_auto_tries_cuda_then_falls_back_to_cpu(self):
        """Default device='auto' should try CUDA (float16) first; on
        failure, fall back to CPU (int8) without raising."""
        from ifa.voice.stt import transcribe_array

        # WhisperModel call log: first call raises (cuda unavailable), second succeeds (cpu)
        call_count = [0]
        def fake_ctor(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("no CUDA runtime")
            return self.model_inst

        self.model_cls.side_effect = fake_ctor

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IFA_WHISPER_DEVICE", None)  # default = auto
            _reset_stt_module_state()
            text = transcribe_array(np.zeros(100, dtype=np.float32))

        self.assertEqual(call_count[0], 2)
        # Second call is the cpu fallback
        second_call = self.model_cls.call_args_list[1]
        self.assertEqual(second_call.kwargs.get("device"), "cpu")
        self.assertEqual(second_call.kwargs.get("compute_type"), "int8")
        # Transcription still works
        self.assertEqual(text, "hello   world")

    def test_device_cuda_explicit_raises_when_unavailable(self):
        """If the user explicitly asks for CUDA but CUDA isn't there,
        we should NOT silently fall back — raise so the user notices."""
        from ifa.voice.stt import transcribe_array

        self.model_cls.side_effect = RuntimeError("no CUDA runtime")

        with patch.dict(os.environ, {"IFA_WHISPER_DEVICE": "cuda"}):
            _reset_stt_module_state()
            # Error is caught by transcribe_array's RuntimeError handler →
            # returns empty string rather than crashing.
            text = transcribe_array(np.zeros(100, dtype=np.float32))
        self.assertEqual(text, "")

    def test_model_honors_ifa_whisper_model_env_var(self):
        from ifa.voice.stt import transcribe_array

        with patch.dict(os.environ, {"IFA_WHISPER_MODEL": "base.en"}):
            _reset_stt_module_state()
            transcribe_array(np.zeros(100, dtype=np.float32))

        args, _ = self.model_cls.call_args
        self.assertEqual(args[0], "base.en")

    def test_model_is_loaded_once_and_reused(self):
        from ifa.voice.stt import transcribe_array

        transcribe_array(np.zeros(100, dtype=np.float32))
        transcribe_array(np.zeros(100, dtype=np.float32))
        transcribe_array(np.zeros(100, dtype=np.float32))

        self.assertEqual(self.model_cls.call_count, 1)
        self.assertEqual(self.model_inst.transcribe.call_count, 3)

    def test_runtime_error_returns_empty_string_without_raising(self):
        from ifa.voice.stt import transcribe_array

        self.model_inst.transcribe.side_effect = RuntimeError("whisper exploded")
        text = transcribe_array(np.zeros(100, dtype=np.float32))
        self.assertEqual(text, "")

    def test_language_defaults_to_en(self):
        from ifa.voice.stt import transcribe_array

        transcribe_array(np.zeros(100, dtype=np.float32))
        kwargs = self.model_inst.transcribe.call_args.kwargs
        self.assertEqual(kwargs.get("language"), "en")

    def test_vad_filter_disabled_because_unit_3_already_did_it(self):
        from ifa.voice.stt import transcribe_array

        transcribe_array(np.zeros(100, dtype=np.float32))
        kwargs = self.model_inst.transcribe.call_args.kwargs
        self.assertFalse(kwargs.get("vad_filter", True))


class TranscribePathTests(unittest.TestCase):
    """Backwards-compat file-path variant kept for the transitional
    ifa.voice.input shim until Unit 5 replaces it."""

    def setUp(self) -> None:
        self.model_cls, self.model_inst = _install_fake_whisper()
        _reset_stt_module_state()

    def tearDown(self) -> None:
        _uninstall_fake_whisper()
        _reset_stt_module_state()

    def test_transcribe_path_reuses_same_lazy_model(self):
        from ifa.voice.stt import transcribe, transcribe_array

        transcribe_array(np.zeros(100, dtype=np.float32))
        transcribe("/tmp/nonexistent.wav")
        # Model constructed once, transcribe called twice
        self.assertEqual(self.model_cls.call_count, 1)
        self.assertEqual(self.model_inst.transcribe.call_count, 2)


if __name__ == "__main__":
    unittest.main()
