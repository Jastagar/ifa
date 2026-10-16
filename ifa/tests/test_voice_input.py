import os
import queue
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _install_fake_sounddevice() -> MagicMock:
    fake = types.ModuleType("sounddevice")
    stream = MagicMock(name="InputStream_instance")
    stream.read.side_effect = lambda n: (np.zeros((n, 1), dtype=np.float32), False)
    stream.read_available = 0
    stream_cls = MagicMock(name="InputStream_class", return_value=stream)
    fake.InputStream = stream_cls
    sys.modules["sounddevice"] = fake
    return stream_cls


def _uninstall_fake_sounddevice() -> None:
    sys.modules.pop("sounddevice", None)


def _install_fake_openwakeword() -> tuple[MagicMock, MagicMock]:
    fake_pkg = types.ModuleType("openwakeword")
    fake_model_mod = types.ModuleType("openwakeword.model")
    fake_utils_mod = types.ModuleType("openwakeword.utils")
    download_mock = MagicMock(name="download_models")
    model_inst = MagicMock(name="Model_instance")
    model_inst.predict = MagicMock(return_value={"ifa": 0.0})
    model_cls = MagicMock(name="Model_class", return_value=model_inst)
    fake_utils_mod.download_models = download_mock
    fake_model_mod.Model = model_cls
    sys.modules["openwakeword"] = fake_pkg
    sys.modules["openwakeword.model"] = fake_model_mod
    sys.modules["openwakeword.utils"] = fake_utils_mod
    return model_cls, model_inst


def _uninstall_fake_openwakeword() -> None:
    for n in ("openwakeword", "openwakeword.model", "openwakeword.utils"):
        sys.modules.pop(n, None)


def _install_fake_whisper() -> MagicMock:
    fake = types.ModuleType("faster_whisper")
    instance = MagicMock(name="whisper_inst")
    segments = [types.SimpleNamespace(text=" hi")]
    instance.transcribe.side_effect = lambda *_a, **_k: (iter(segments), types.SimpleNamespace())
    cls = MagicMock(name="WhisperModel", return_value=instance)
    fake.WhisperModel = cls
    sys.modules["faster_whisper"] = fake
    import ifa.voice.stt as stt
    stt._model = None
    return cls


def _uninstall_fake_whisper() -> None:
    sys.modules.pop("faster_whisper", None)
    import ifa.voice.stt as stt
    stt._model = None


class _FakeTTS:
    @property
    def is_speaking(self) -> bool:
        return False


class InitInputTests(unittest.TestCase):
    def test_text_mode_is_default(self):
        from ifa.voice.input import _TextMode, init_input

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IFA_MODE", None)
            mode = init_input(tts_service=_FakeTTS())
        self.assertIsInstance(mode, _TextMode)

    def test_text_mode_arm_followup_is_noop(self):
        from ifa.voice.input import init_input

        with patch.dict(os.environ, {"IFA_MODE": "text"}):
            mode = init_input(tts_service=_FakeTTS())
        mode.arm_followup()
        self.assertTrue(hasattr(mode, "get"))

    def test_voice_mode_init_failure_falls_back_to_text(self):
        from ifa.voice.input import _TextMode, init_input

        with patch.dict(os.environ, {"IFA_MODE": "voice"}), \
             patch("ifa.voice.input.VoiceInput", side_effect=RuntimeError("no mic")):
            mode = init_input(tts_service=_FakeTTS())
        self.assertIsInstance(mode, _TextMode)


class VoiceInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sd_cls = _install_fake_sounddevice()
        self.ow_cls, self.ow_inst = _install_fake_openwakeword()
        self.whisper_cls = _install_fake_whisper()

    def tearDown(self) -> None:
        _uninstall_fake_whisper()
        _uninstall_fake_openwakeword()
        _uninstall_fake_sounddevice()

    def _make_voice_input(self, *, wake_returns=None, capture_audio=None, transcript="hi"):
        from ifa.voice.input import VoiceInput

        if wake_returns is None:
            wake_returns = [0.9]
        if capture_audio is None:
            capture_audio = np.zeros(16_000, dtype=np.float32)

        wake_iter = iter(wake_returns)

        def fake_wait(read_chunk):
            try:
                return next(wake_iter)
            except StopIteration:
                raise _LoopStop()

        vi = VoiceInput(tts_service=_FakeTTS())
        vi._listener.wait_for_wake = fake_wait
        vi._capture_utterance = MagicMock(return_value=capture_audio)
        vi._transcribe = MagicMock(return_value=transcript)
        vi.start()
        return vi

    def test_happy_path_queues_one_transcribed_utterance(self):
        vi = self._make_voice_input(
            wake_returns=[0.9],
            transcript="set a timer for five minutes",
        )
        try:
            got = vi.get()
            self.assertEqual(got, "set a timer for five minutes")
        finally:
            vi.close()

    def test_empty_transcript_is_not_queued(self):
        vi = self._make_voice_input(
            wake_returns=[0.9, 0.9],
            transcript="",
        )
        try:
            time.sleep(0.2)
            with self.assertRaises(queue.Empty):
                vi._queue.get(timeout=0.1)
        finally:
            vi.close()

    def test_loop_exception_does_not_kill_thread(self):
        from ifa.voice.input import VoiceInput

        vi = VoiceInput(tts_service=_FakeTTS())

        call_count = [0]

        def fake_wait(read_chunk):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom")
            if call_count[0] == 2:
                return 0.9
            raise _LoopStop()

        vi._listener.wait_for_wake = fake_wait
        vi._capture_utterance = MagicMock(return_value=np.zeros(100, dtype=np.float32))
        vi._transcribe = MagicMock(return_value="recovered")
        vi.start()
        try:
            got = vi.get()
            self.assertEqual(got, "recovered")
        finally:
            vi.close()

    def test_mic_stream_is_opened_at_16k_mono_float32(self):
        vi = self._make_voice_input()
        try:
            self.sd_cls.assert_called_once()
            kwargs = self.sd_cls.call_args.kwargs
            self.assertEqual(kwargs["samplerate"], 16_000)
            self.assertEqual(kwargs["channels"], 1)
            self.assertEqual(kwargs["dtype"], "float32")
        finally:
            vi.close()

    def test_close_stops_the_stream(self):
        vi = self._make_voice_input()
        vi.close()
        vi._stream.stop.assert_called()
        vi._stream.close.assert_called()


class _LoopStop(BaseException):
    pass


class OrchestratorInitInputIntegrationTests(unittest.TestCase):
    def test_orchestrator_uses_input_mode_and_arms_followup_each_turn(self):
        from ifa.core import orchestrator

        inputs = iter(["hello", "exit"])
        fake_mode = MagicMock(name="input_mode")
        fake_mode.get.side_effect = lambda: next(inputs)

        init_input_calls = []
        def fake_init_input(tts_service):
            init_input_calls.append(tts_service)
            return fake_mode

        with patch("ifa.core.orchestrator.check_health"), \
             patch("ifa.core.orchestrator.load_n8n_config", return_value={}), \
             patch("ifa.core.orchestrator.init_db"), \
             patch("ifa.core.orchestrator.register_all"), \
             patch("ifa.core.orchestrator.resume_reminders"), \
             patch("ifa.core.orchestrator.TTSService") as tts_cls, \
             patch("ifa.core.orchestrator.init_input", side_effect=fake_init_input), \
             patch("ifa.core.orchestrator.agent_turn", return_value="hi back"):
            orchestrator.run()

        self.assertEqual(len(init_input_calls), 1)
        self.assertIs(init_input_calls[0], tts_cls.return_value)
        self.assertEqual(fake_mode.arm_followup.call_count, 1)


class VoiceInputFollowupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sd_cls = _install_fake_sounddevice()
        self.ow_cls, self.ow_inst = _install_fake_openwakeword()
        self.whisper_cls = _install_fake_whisper()

    def tearDown(self) -> None:
        _uninstall_fake_whisper()
        _uninstall_fake_openwakeword()
        _uninstall_fake_sounddevice()

    def test_arm_followup_sets_deadline_in_future(self):
        from ifa.voice.input import VoiceInput

        vi = VoiceInput(tts_service=_FakeTTS())
        try:
            self.assertFalse(vi._in_followup_window())
            vi.arm_followup()
            self.assertTrue(vi._in_followup_window())
        finally:
            vi.close()

    def test_arm_followup_is_noop_when_window_is_zero(self):
        from ifa.voice.input import VoiceInput

        with patch.dict(os.environ, {"IFA_FOLLOWUP_WINDOW_SEC": "0"}):
            vi = VoiceInput(tts_service=_FakeTTS())
        try:
            vi.arm_followup()
            self.assertFalse(vi._in_followup_window())
        finally:
            vi.close()

    def test_followup_bypasses_wake_word_and_passes_start_timeout(self):
        from ifa.voice.input import VoiceInput

        vi = VoiceInput(tts_service=_FakeTTS())
        vi._followup_window_sec = 5.0
        vi.arm_followup()

        wait_called = [False]
        def fake_wait(read_chunk):
            wait_called[0] = True
            raise _LoopStop()

        capture_kwargs_seen = {}
        def fake_capture(read_chunk, **kwargs):
            capture_kwargs_seen.update(kwargs)
            raise _LoopStop()

        vi._listener.wait_for_wake = fake_wait
        vi._capture_utterance = fake_capture
        vi._transcribe = MagicMock(return_value="")
        vi.start()
        vi._thread.join(timeout=1.0)
        try:
            self.assertFalse(wait_called[0], "wait_for_wake must be skipped during follow-up")
            self.assertEqual(capture_kwargs_seen.get("start_timeout_ms"), 5000)
        finally:
            vi.close()

    def test_capture_blocks_until_tts_is_silent(self):
        from ifa.voice.input import VoiceInput

        class _SpeakingThenSilentTTS:
            def __init__(self):
                self._calls = 0
            @property
            def is_speaking(self):
                self._calls += 1
                return self._calls < 3

        speaking_tts = _SpeakingThenSilentTTS()
        vi = VoiceInput(tts_service=speaking_tts)

        capture_started_at_call = [None]
        def fake_capture(read_chunk, **kwargs):
            capture_started_at_call[0] = speaking_tts._calls
            raise _LoopStop()

        vi._listener.wait_for_wake = lambda read_chunk: None
        vi._capture_utterance = fake_capture
        vi._transcribe = MagicMock(return_value="")
        vi.start()
        vi._thread.join(timeout=2.0)
        try:
            self.assertIsNotNone(capture_started_at_call[0])
            self.assertGreaterEqual(capture_started_at_call[0], 3)
        finally:
            vi.close()

    def test_voice_loop_waits_for_arm_followup_before_next_decision(self):
        from ifa.voice.input import VoiceInput

        vi = VoiceInput(tts_service=_FakeTTS())

        wait_started_iteration_2 = threading.Event()

        iteration_count = [0]
        def fake_wait_for_wake(read_chunk):
            iteration_count[0] += 1
            if iteration_count[0] == 1:
                return 0.9
            wait_started_iteration_2.set()
            raise _LoopStop()

        vi._listener.wait_for_wake = fake_wait_for_wake
        vi._capture_utterance = MagicMock(return_value=np.zeros(100, dtype=np.float32))
        vi._transcribe = MagicMock(return_value="hello")
        vi.start()
        try:
            self.assertEqual(vi.get(), "hello")
            time.sleep(0.2)
            self.assertFalse(
                wait_started_iteration_2.is_set(),
                "voice loop iterated to wait_for_wake without waiting for arm_followup",
            )
            vi.arm_followup()
            time.sleep(0.2)
        finally:
            vi.close()

    def test_arm_followup_releases_blocked_voice_thread_via_event(self):
        from ifa.voice.input import VoiceInput

        with patch.dict(os.environ, {"IFA_FOLLOWUP_WINDOW_SEC": "0"}):
            vi = VoiceInput(tts_service=_FakeTTS())
        try:
            vi._turn_complete.clear()
            self.assertFalse(vi._turn_complete.is_set())
            vi.arm_followup()
            self.assertTrue(vi._turn_complete.is_set())
        finally:
            vi.close()

    def test_drain_stream_called_before_each_capture(self):
        from ifa.voice.input import VoiceInput

        vi = VoiceInput(tts_service=_FakeTTS())

        order: list[str] = []
        original_drain = vi._drain_stream
        def spy_drain():
            order.append("drain")
            return original_drain()

        def fake_capture(read_chunk, **kwargs):
            order.append("capture")
            raise _LoopStop()

        vi._drain_stream = spy_drain
        vi._listener.wait_for_wake = lambda read_chunk: None
        vi._capture_utterance = fake_capture
        vi._transcribe = MagicMock(return_value="")
        vi.start()
        vi._thread.join(timeout=2.0)
        try:
            self.assertEqual(order, ["drain", "capture"])
        finally:
            vi.close()

    def test_listening_cue_plays_before_capture(self):
        from ifa.voice.input import VoiceInput

        vi = VoiceInput(tts_service=_FakeTTS())
        order: list[str] = []

        def fake_cue():
            order.append("cue")

        def fake_capture(read_chunk, **kwargs):
            order.append("capture")
            raise _LoopStop()

        vi._play_listening_cue = fake_cue
        vi._listener.wait_for_wake = lambda read_chunk: None
        vi._capture_utterance = fake_capture
        vi.start()
        vi._thread.join(timeout=2.0)
        try:
            self.assertEqual(order, ["cue", "capture"])
        finally:
            vi.close()

    def test_followup_capture_is_silent(self):
        from ifa.voice.input import VoiceInput

        vi = VoiceInput(tts_service=_FakeTTS())
        cues: list[str] = []
        vi._in_followup_window = lambda: True
        vi._play_listening_cue = lambda: cues.append("cue")

        def fake_capture(read_chunk, **kwargs):
            raise _LoopStop()

        vi._capture_utterance = fake_capture
        vi.start()
        vi._thread.join(timeout=2.0)
        try:
            self.assertEqual(cues, [])
        finally:
            vi.close()


if __name__ == "__main__":
    unittest.main()
