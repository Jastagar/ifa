"""Tests for ifa.services.tts_service.TTSService.

Run: python -m unittest ifa.tests.test_tts_service -v
"""
import os
import subprocess
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import sounddevice as sd

from ifa.services.tts_service import TTSService


def _fake_engine_that_writes_wav():
    """Returns a pyttsx3.init() replacement that writes a short silent WAV
    on runAndWait()."""
    state = {"target_path": None}
    engine = MagicMock()

    def save_to_file(text, path):
        state["target_path"] = path

    def run_and_wait():
        if state["target_path"]:
            import soundfile as sf
            silence = np.zeros(800, dtype="float32")
            sf.write(state["target_path"], silence, 16000)

    engine.save_to_file.side_effect = save_to_file
    engine.runAndWait.side_effect = run_and_wait
    return engine


class TTSServiceMainThreadTests(unittest.TestCase):
    def test_empty_text_is_noop(self):
        tts = TTSService()
        with patch("ifa.services.tts_service.pyttsx3.init") as init, \
             patch("ifa.services.tts_service.sd.play") as play:
            tts.speak("")
            init.assert_not_called()
            play.assert_not_called()

    def test_speak_on_main_thread_plays_on_default(self):
        tts = TTSService()
        with patch("ifa.services.tts_service.pyttsx3.init",
                   return_value=_fake_engine_that_writes_wav()), \
             patch("ifa.services.tts_service.sd.play") as play, \
             patch("ifa.services.tts_service.sd.wait"):
            tts.speak("hello")
            self.assertEqual(play.call_count, 1)
            _args, kwargs = play.call_args
            self.assertIsNone(kwargs.get("device"))

    def test_port_audio_error_on_play_does_not_raise(self):
        tts = TTSService()
        with patch("ifa.services.tts_service.pyttsx3.init",
                   return_value=_fake_engine_that_writes_wav()), \
             patch("ifa.services.tts_service.sd.play",
                   side_effect=sd.PortAudioError("boom")), \
             patch("ifa.services.tts_service.sd.wait"):
            tts.speak("hello")  # must not raise

    def test_port_audio_error_on_wait_does_not_raise(self):
        tts = TTSService()
        with patch("ifa.services.tts_service.pyttsx3.init",
                   return_value=_fake_engine_that_writes_wav()), \
             patch("ifa.services.tts_service.sd.play"), \
             patch("ifa.services.tts_service.sd.wait",
                   side_effect=sd.PortAudioError("boom")):
            tts.speak("hello")  # must not raise

    def test_temp_wav_is_cleaned_up(self):
        tts = TTSService()
        captured = {}

        def fake_init():
            engine = _fake_engine_that_writes_wav()
            orig_save = engine.save_to_file.side_effect

            def track(text, path):
                captured["path"] = path
                orig_save(text, path)

            engine.save_to_file.side_effect = track
            return engine

        with patch("ifa.services.tts_service.pyttsx3.init", side_effect=fake_init), \
             patch("ifa.services.tts_service.sd.play"), \
             patch("ifa.services.tts_service.sd.wait"):
            tts.speak("hello")

        self.assertIn("path", captured)
        self.assertFalse(os.path.exists(captured["path"]))

    def test_empty_wav_skips_playback(self):
        """If pyttsx3 writes an empty file, play() is not invoked."""
        tts = TTSService()
        engine = MagicMock()
        engine.save_to_file.side_effect = lambda text, path: open(path, "w").close()
        engine.runAndWait.side_effect = lambda: None

        with patch("ifa.services.tts_service.pyttsx3.init", return_value=engine), \
             patch("ifa.services.tts_service.sd.play") as play:
            tts.speak("hello")
            play.assert_not_called()


class TTSServiceCrossThreadTests(unittest.TestCase):
    def test_cross_thread_call_queues_and_drains(self):
        tts = TTSService()
        calls = []

        def fake_speak_on_main(text):
            calls.append(text)

        with patch.object(tts, "_speak_on_main", side_effect=fake_speak_on_main):
            done_flag = threading.Event()

            def worker():
                tts.speak("from thread")
                done_flag.set()

            t = threading.Thread(target=worker)
            t.start()

            # Worker is blocked until drain_queue runs on the main thread.
            self.assertFalse(done_flag.wait(timeout=0.1))
            self.assertEqual(calls, [])

            tts.drain_queue()

            self.assertTrue(done_flag.wait(timeout=1.0))
            self.assertEqual(calls, ["from thread"])
            t.join(timeout=1.0)

    def test_drain_queue_when_empty_is_noop(self):
        tts = TTSService()
        tts.drain_queue()  # must not block or raise

    def test_main_thread_call_does_not_use_queue(self):
        tts = TTSService()
        with patch.object(tts, "_speak_on_main") as fake_main:
            tts.speak("direct")
            fake_main.assert_called_once_with("direct")
        self.assertTrue(tts._queue.empty())


class ManagerDecouplingTests(unittest.TestCase):
    """Plan O1: manager.py must not construct its own TTSService."""

    def test_manager_does_not_instantiate_tts_service(self):
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        result = subprocess.run(
            ["grep", "-n", "TTSService()", "ifa/skills/manager.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout,
            "",
            f"manager.py constructs TTSService() -- should receive it via handle_with_intent parameter. Found: {result.stdout!r}",
        )


if __name__ == "__main__":
    unittest.main()
