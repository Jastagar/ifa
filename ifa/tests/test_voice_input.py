"""Tests for ifa.voice.input.VoiceInput + init_input.

sounddevice, openWakeWord, Silero VAD, and Whisper are all mocked so
tests run without real audio hardware or model files.

Run: PYTHONPATH=. python -m unittest ifa.tests.test_voice_input -v
"""
import os
import queue
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------- fake module installers ----------


def _install_fake_sounddevice() -> MagicMock:
    fake = types.ModuleType("sounddevice")
    stream = MagicMock(name="InputStream_instance")
    # default: return zero-valued chunks for any read size
    stream.read.side_effect = lambda n: (np.zeros((n, 1), dtype=np.float32), False)
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
    model_inst.predict = MagicMock(return_value={"hey_mycroft": 0.0})
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
    """Replace faster_whisper.WhisperModel with a mock for the STT module."""
    fake = types.ModuleType("faster_whisper")
    instance = MagicMock(name="whisper_inst")
    # default: return ("hi",) segments + info
    segments = [types.SimpleNamespace(text=" hi")]
    instance.transcribe.side_effect = lambda *_a, **_k: (iter(segments), types.SimpleNamespace())
    cls = MagicMock(name="WhisperModel", return_value=instance)
    fake.WhisperModel = cls
    sys.modules["faster_whisper"] = fake
    # Force the STT module's lazy cache to reload
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


# ---------- init_input tests ----------


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
        # Must not raise — orchestrator calls this after every turn.
        mode.arm_followup()
        self.assertTrue(hasattr(mode, "get"))

    def test_voice_mode_init_failure_falls_back_to_text(self):
        """If VoiceInput init raises (e.g. no mic), we return the text
        mode stub rather than crashing the orchestrator."""
        from ifa.voice.input import _TextMode, init_input

        with patch.dict(os.environ, {"IFA_MODE": "voice"}), \
             patch("ifa.voice.input.VoiceInput", side_effect=RuntimeError("no mic")):
            mode = init_input(tts_service=_FakeTTS())
        self.assertIsInstance(mode, _TextMode)


# ---------- VoiceInput tests (real class, mocked deps) ----------


class VoiceInputTests(unittest.TestCase):
    """Exercise VoiceInput with every external dep mocked.

    NOTE: VoiceInput spawns a daemon thread on construction. Tests
    should keep that thread well-scripted (mock ``wait_for_wake``,
    ``capture_utterance``, ``transcribe_array``) so it does not block
    indefinitely on unmocked mic reads.
    """

    def setUp(self) -> None:
        self.sd_cls = _install_fake_sounddevice()
        self.ow_cls, self.ow_inst = _install_fake_openwakeword()
        self.whisper_cls = _install_fake_whisper()

    def tearDown(self) -> None:
        _uninstall_fake_whisper()
        _uninstall_fake_openwakeword()
        _uninstall_fake_sounddevice()

    def _make_voice_input(self, *, wake_returns=None, capture_audio=None, transcript="hi"):
        """Construct a VoiceInput whose inner components are MagicMocks.

        Overrides the per-instance callables BEFORE starting the daemon
        thread, so the loop picks up mocks from iteration zero.

        - wake_returns: list of values to return from wait_for_wake; after
          exhaustion, raises to end the loop.
        - capture_audio: numpy array returned by capture_utterance.
        - transcript: string returned by transcribe_array.
        """
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
        # Replace before start() so the daemon thread never sees unmocked deps.
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
        """VAD fired but Whisper returned nothing → loop back, don't
        surface an empty string to the agent."""
        vi = self._make_voice_input(
            wake_returns=[0.9, 0.9],  # two wake events
            transcript="",
        )
        try:
            # After 0.2s, still empty — queue stays empty
            time.sleep(0.2)
            with self.assertRaises(queue.Empty):
                vi._queue.get(timeout=0.1)
        finally:
            vi.close()

    def test_loop_exception_does_not_kill_thread(self):
        """A transient per-turn error shouldn't take voice mode down
        for the rest of the session."""
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
    """Sentinel used to end the daemon loop in tests without blocking.

    Inherits from BaseException (not Exception) so VoiceInput's
    ``except BaseException`` branch lets it propagate and exit the
    thread cleanly — and, critically, no "loop error:" log is emitted.
    """


# ---------- Orchestrator plumbing test ----------


class OrchestratorInitInputIntegrationTests(unittest.TestCase):
    """Smoke-check that orchestrator.run wires init_input correctly.

    Only the plumbing path is exercised: agent_turn and check_health are
    mocked out, and the input mode is a canned stub.
    """

    def test_orchestrator_uses_input_mode_and_arms_followup_each_turn(self):
        """orchestrator.run must (a) call init_input(tts), (b) loop on
        input_mode.get(), (c) call arm_followup() after every tts.speak()."""
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
        # arm_followup was called once — after the one real turn ("hello"),
        # NOT after the "exit" command (that short-circuits the loop).
        self.assertEqual(fake_mode.arm_followup.call_count, 1)


class VoiceInputFollowupTests(unittest.TestCase):
    """Exercise the follow-up-window behavior that skips wake-word after a reply."""

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
        """When armed, the loop skips wait_for_wake and calls capture
        with ``start_timeout_ms`` matching the follow-up window."""
        from ifa.voice.input import VoiceInput

        vi = VoiceInput(tts_service=_FakeTTS())
        vi._followup_window_sec = 5.0
        vi.arm_followup()  # arm BEFORE starting thread

        wait_called = [False]
        def fake_wait(read_chunk):
            wait_called[0] = True
            raise _LoopStop()  # should NOT be called in follow-up path

        capture_kwargs_seen = {}
        def fake_capture(read_chunk, **kwargs):
            capture_kwargs_seen.update(kwargs)
            raise _LoopStop()  # end after first capture

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


if __name__ == "__main__":
    unittest.main()
