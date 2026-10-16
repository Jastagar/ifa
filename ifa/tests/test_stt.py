import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _install_fake_whisper() -> tuple[MagicMock, MagicMock]:
    fake = types.ModuleType("faster_whisper")
    model_instance = MagicMock(name="whisper_instance")

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
        self.assertEqual(text, "hello   world")

    def test_empty_audio_returns_empty_string_without_loading_model(self):
        from ifa.voice.stt import transcribe_array

        text = transcribe_array(np.zeros(0, dtype=np.float32))
        self.assertEqual(text, "")
        self.model_cls.assert_not_called()

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
        from ifa.voice.stt import transcribe_array

        with patch.dict(os.environ, {"IFA_WHISPER_DEVICE": "cpu"}):
            _reset_stt_module_state()
            transcribe_array(np.zeros(100, dtype=np.float32))

        args, kwargs = self.model_cls.call_args
        self.assertEqual(args[0], "small.en")
        self.assertNotIn("size", kwargs)
        self.assertEqual(kwargs.get("compute_type"), "int8")
        self.assertEqual(kwargs.get("device"), "cpu")

    def test_device_auto_tries_cuda_then_falls_back_to_cpu(self):
        from ifa.voice.stt import transcribe_array

        call_count = [0]
        def fake_ctor(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("no CUDA runtime")
            return self.model_inst

        self.model_cls.side_effect = fake_ctor

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IFA_WHISPER_DEVICE", None)
            os.environ.pop("IFA_WHISPER_COMPUTE_TYPE", None)
            _reset_stt_module_state()
            text = transcribe_array(np.zeros(100, dtype=np.float32))

        self.assertEqual(call_count[0], 2)
        first_call = self.model_cls.call_args_list[0]
        self.assertEqual(first_call.kwargs.get("device"), "auto")
        self.assertEqual(first_call.kwargs.get("compute_type"), "int8_float16")
        second_call = self.model_cls.call_args_list[1]
        self.assertEqual(second_call.kwargs.get("device"), "cpu")
        self.assertEqual(second_call.kwargs.get("compute_type"), "int8")
        self.assertEqual(text, "hello   world")

    def test_compute_type_env_override_applies_to_gpu_path(self):
        from ifa.voice.stt import transcribe_array

        with patch.dict(
            os.environ,
            {
                "IFA_WHISPER_DEVICE": "cuda",
                "IFA_WHISPER_COMPUTE_TYPE": "float16",
            },
        ):
            _reset_stt_module_state()
            transcribe_array(np.zeros(100, dtype=np.float32))

        kwargs = self.model_cls.call_args.kwargs
        self.assertEqual(kwargs.get("device"), "cuda")
        self.assertEqual(kwargs.get("compute_type"), "float16")

    def test_device_cuda_explicit_raises_when_unavailable(self):
        from ifa.voice.stt import transcribe_array

        self.model_cls.side_effect = RuntimeError("no CUDA runtime")

        with patch.dict(os.environ, {"IFA_WHISPER_DEVICE": "cuda"}):
            _reset_stt_module_state()
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


if __name__ == "__main__":
    unittest.main()
