import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _install_fake_openwakeword() -> tuple[MagicMock, MagicMock, MagicMock]:
    fake_pkg = types.ModuleType("openwakeword")
    fake_model_mod = types.ModuleType("openwakeword.model")
    fake_utils_mod = types.ModuleType("openwakeword.utils")

    download_mock = MagicMock(name="download_models")
    model_instance = MagicMock(name="Model_instance")
    model_instance.predict = MagicMock(return_value={"ifa": 0.0})
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
    def __init__(self, speaking: bool = False) -> None:
        self._speaking = speaking

    @property
    def is_speaking(self) -> bool:
        return self._speaking


class WakeWordListenerInitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.download, self.model_cls, self.model_inst = _install_fake_openwakeword()
        self.model_inst.predict.return_value = {"ifa": 0.0}

    def tearDown(self) -> None:
        _uninstall_fake_openwakeword()

    def test_init_resolves_default_to_bundled_ifa_path_no_download(self):
        from ifa.voice.wake_word import WakeWordListener

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IFA_WAKE_MODEL", None)
            listener = WakeWordListener(tts_service=None)

        self.assertEqual(listener.score_key, "ifa")
        self.assertIsNone(listener.fallback_from)
        self.download.assert_not_called()
        self.model_cls.assert_called_once()
        kwargs = self.model_cls.call_args.kwargs
        wakeword_models = kwargs["wakeword_models"]
        self.assertEqual(len(wakeword_models), 1)
        self.assertTrue(
            wakeword_models[0].endswith("models/ifa.onnx")
            or wakeword_models[0].endswith("models\\ifa.onnx"),
            f"expected wakeword_models to end with models/ifa.onnx, got {wakeword_models[0]!r}",
        )
        self.assertEqual(kwargs["inference_framework"], "onnx")

    def test_ifa_wake_model_env_override_selects_builtin(self):
        from ifa.voice.wake_word import WakeWordListener

        with patch.dict(os.environ, {"IFA_WAKE_MODEL": "alexa"}):
            listener = WakeWordListener(tts_service=None)

        self.assertEqual(listener.score_key, "alexa")
        self.download.assert_called_once_with(model_names=["alexa"])
        self.assertEqual(
            self.model_cls.call_args.kwargs["wakeword_models"], ["alexa"]
        )

    def test_custom_onnx_path_skips_download_and_derives_score_key(self):
        from ifa.voice.wake_word import WakeWordListener

        import pathlib
        existing = str(pathlib.Path(__file__).resolve())
        fake_custom = existing

        with patch.dict(os.environ, {"IFA_WAKE_MODEL": fake_custom}):
            listener = WakeWordListener(tts_service=None)

        self.download.assert_not_called()
        self.assertEqual(
            self.model_cls.call_args.kwargs["wakeword_models"], [fake_custom]
        )
        expected_key = pathlib.Path(existing).stem
        self.assertEqual(listener.score_key, expected_key)

    def test_model_spec_kwarg_overrides_env(self):
        from ifa.voice.wake_word import WakeWordListener

        with patch.dict(os.environ, {"IFA_WAKE_MODEL": "alexa"}):
            listener = WakeWordListener(tts_service=None, model_spec="hey_jarvis")
        self.assertEqual(listener.score_key, "hey_jarvis")

    def test_path_spec_falls_back_to_built_in_when_file_missing(self):
        import io
        import contextlib
        from ifa.voice.wake_word import WakeWordListener

        self.model_inst.predict.return_value = {"ifa": 0.0}

        missing_path = "/nonexistent/dir/ifa.onnx"
        captured = io.StringIO()
        with patch.dict(os.environ, {"IFA_WAKE_MODEL": missing_path}), \
             contextlib.redirect_stdout(captured):
            listener = WakeWordListener(tts_service=None)

        self.assertEqual(listener.score_key, "ifa")
        self.assertEqual(listener.fallback_from, missing_path)
        self.download.assert_called_once_with(model_names=["ifa"])
        self.assertEqual(
            self.model_cls.call_args.kwargs["wakeword_models"], ["ifa"]
        )
        out = captured.getvalue()
        self.assertIn("WARNING", out)
        self.assertIn(missing_path, out)
        self.assertIn("ifa", out)

    def test_no_fallback_when_path_spec_actually_exists(self):
        from ifa.voice.wake_word import WakeWordListener
        import pathlib

        existing = str(pathlib.Path(__file__).resolve())
        with patch.dict(os.environ, {"IFA_WAKE_MODEL": existing}):
            listener = WakeWordListener(tts_service=None)

        self.assertIsNone(listener.fallback_from)
        self.download.assert_not_called()
        self.assertEqual(
            self.model_cls.call_args.kwargs["wakeword_models"], [existing]
        )

    def test_no_fallback_when_built_in_name_used(self):
        from ifa.voice.wake_word import WakeWordListener

        self.model_inst.predict.return_value = {"alexa": 0.0}
        with patch.dict(os.environ, {"IFA_WAKE_MODEL": "alexa"}):
            listener = WakeWordListener(tts_service=None)

        self.assertIsNone(listener.fallback_from)
        self.assertEqual(listener.score_key, "alexa")
        self.download.assert_called_once_with(model_names=["alexa"])

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
        with patch.dict(os.environ, {"IFA_WAKE_MODEL": "ifa"}):
            with self.assertRaises(WakeWordInitError) as cm:
                WakeWordListener(tts_service=None)
        msg = str(cm.exception)
        self.assertIn("ifa", msg)
        self.assertIn("internet", msg.lower())


class WakeWordListenerDetectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.download, self.model_cls, self.model_inst = _install_fake_openwakeword()

    def tearDown(self) -> None:
        _uninstall_fake_openwakeword()

    def _make_read_chunk(self, n_silent: int):
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
        from ifa.voice.wake_word import WakeWordListener

        self.model_inst.predict.side_effect = [
            {"ifa": 0.1},
            {"ifa": 0.4},
            {"ifa": 0.8},
            {"ifa": 0.9},
        ]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IFA_WAKE_CONSECUTIVE", None)
            listener = WakeWordListener(tts_service=None, threshold=0.5)

        read, frames_yielded = self._make_read_chunk(n_silent=10)
        score = listener.wait_for_wake(read_chunk=read)
        self.assertEqual(score, 0.9)
        self.assertEqual(frames_yielded[0], 4)
        self.assertEqual(self.model_inst.predict.call_count, 4)

    def test_single_frame_spike_does_not_fire_with_default_consecutive(self):
        from ifa.voice.wake_word import WakeWordListener

        self.model_inst.predict.side_effect = [
            {"ifa": 0.9},
            {"ifa": 0.1},
            {"ifa": 0.9},
            {"ifa": 0.9},
        ]

        listener = WakeWordListener(tts_service=None, threshold=0.5)
        read, _ = self._make_read_chunk(n_silent=10)
        listener.wait_for_wake(read_chunk=read)
        self.assertEqual(self.model_inst.predict.call_count, 4)

    def test_consecutive_env_override_to_one_restores_single_frame_firing(self):
        from ifa.voice.wake_word import WakeWordListener

        self.model_inst.predict.side_effect = [
            {"ifa": 0.9},
        ]
        with patch.dict(os.environ, {"IFA_WAKE_CONSECUTIVE": "1"}):
            listener = WakeWordListener(tts_service=None, threshold=0.5)
        read, _ = self._make_read_chunk(n_silent=10)
        listener.wait_for_wake(read_chunk=read)
        self.assertEqual(self.model_inst.predict.call_count, 1)

    def test_tts_speaking_resets_consecutive_streak(self):
        from ifa.voice.wake_word import WakeWordListener

        tts = _FakeTTS(speaking=False)
        listener = WakeWordListener(tts_service=tts, threshold=0.5)

        self.model_inst.predict.side_effect = [
            {"ifa": 0.9},
            {"ifa": 0.0},
            {"ifa": 0.9},
            {"ifa": 0.9},
        ]

        call_counter = [0]
        def read():
            call_counter[0] += 1
            tts._speaking = call_counter[0] == 2
            return np.zeros(1280, dtype=np.float32)

        listener.wait_for_wake(read_chunk=read)
        self.assertEqual(call_counter[0], 4)
        self.assertEqual(self.model_inst.predict.call_count, 4)

    def test_mic_chunks_replaced_with_silence_during_tts_speaking(self):
        from ifa.voice.wake_word import WakeWordListener

        tts = _FakeTTS(speaking=True)
        with patch.dict(os.environ, {"IFA_WAKE_CONSECUTIVE": "1"}):
            listener = WakeWordListener(tts_service=tts, threshold=0.5)

        self.model_inst.predict.return_value = {"ifa": 0.9}

        call_counter = [0]

        def read_then_unmute():
            if call_counter[0] == 3:
                tts._speaking = False
            call_counter[0] += 1
            return np.full(1280, 0.5, dtype=np.float32)

        listener.wait_for_wake(read_chunk=read_then_unmute)

        self.assertEqual(call_counter[0], 4)
        self.assertEqual(self.model_inst.predict.call_count, 4)

        muted_calls = self.model_inst.predict.call_args_list[:3]
        for call in muted_calls:
            arr = call[0][0]
            self.assertEqual(arr.dtype, np.int16)
            self.assertTrue(
                np.all(arr == 0),
                "muted predict must receive zeros, not the live mic chunk",
            )

        live_call_arr = self.model_inst.predict.call_args_list[3][0][0]
        self.assertTrue(
            np.any(live_call_arr != 0),
            "post-unmute predict must receive the real mic chunk",
        )

    def test_model_reset_called_after_successful_detection(self):
        from ifa.voice.wake_word import WakeWordListener

        with patch.dict(os.environ, {"IFA_WAKE_CONSECUTIVE": "1"}):
            listener = WakeWordListener(tts_service=None, threshold=0.5)
        self.model_inst.predict.return_value = {"ifa": 0.9}

        listener.wait_for_wake(
            read_chunk=lambda: np.zeros(1280, dtype=np.float32)
        )
        self.model_inst.reset.assert_called_once()

    def test_int16_input_passes_through_without_reconversion(self):
        from ifa.voice.wake_word import WakeWordListener

        listener = WakeWordListener(tts_service=None, threshold=0.5)
        self.model_inst.predict.return_value = {"ifa": 0.9}

        int16_chunk = np.zeros(1280, dtype=np.int16)
        listener.wait_for_wake(read_chunk=lambda: int16_chunk)

        passed = self.model_inst.predict.call_args[0][0]
        self.assertEqual(passed.dtype, np.int16)

    def test_float32_input_converted_to_int16_pcm_range(self):
        from ifa.voice.wake_word import WakeWordListener

        listener = WakeWordListener(tts_service=None, threshold=0.5)
        self.model_inst.predict.return_value = {"ifa": 0.9}

        float_chunk = np.linspace(-1.0, 1.0, 1280, dtype=np.float32)
        listener.wait_for_wake(read_chunk=lambda: float_chunk)

        passed = self.model_inst.predict.call_args[0][0]
        self.assertEqual(passed.dtype, np.int16)
        self.assertEqual(passed.min(), -32768)
        self.assertLessEqual(passed.max(), 32767)
        self.assertGreaterEqual(passed.max(), 32700)

    def test_missing_score_key_treated_as_zero(self):
        from ifa.voice.wake_word import WakeWordListener

        with patch.dict(os.environ, {"IFA_WAKE_CONSECUTIVE": "1"}):
            listener = WakeWordListener(tts_service=None, threshold=0.5)
        self.model_inst.predict.side_effect = [
            {},
            {"ifa": 0.9},
        ]
        call_counter = [0]

        def read():
            call_counter[0] += 1
            return np.zeros(1280, dtype=np.float32)

        listener.wait_for_wake(read_chunk=read)
        self.assertEqual(call_counter[0], 2)


class BundledOnnxIntegrityTests(unittest.TestCase):
    def test_bundled_ifa_onnx_exists(self):
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
        path = repo_root / "ifa" / "models" / "ifa.onnx"
        self.assertTrue(
            path.exists(),
            f"ifa/models/ifa.onnx must exist for the default wake-word model "
            f"to load. Expected at: {path}",
        )

    def test_bundled_ifa_onnx_is_loadable_via_onnxruntime(self):
        import pathlib
        import onnxruntime

        repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
        path = repo_root / "ifa" / "models" / "ifa.onnx"
        if not path.exists():
            self.skipTest("ifa.onnx not present (separate test asserts existence)")

        sess = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.assertGreater(len(sess.get_inputs()), 0)
        self.assertGreater(len(sess.get_outputs()), 0)


if __name__ == "__main__":
    unittest.main()
