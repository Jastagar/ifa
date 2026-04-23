"""Tests for ifa.services.tts_service.TTSService.

Run: python -m unittest ifa.tests.test_tts_service -v
"""
import ast
import base64
import pathlib
import threading
import unittest
from unittest.mock import patch

from ifa.services.tts_service import TTSService


def _mock_tempfile():
    """Patch tempfile so tests never touch the real filesystem."""
    return patch(
        "ifa.services.tts_service.tempfile.mkstemp",
        return_value=(9999, "/tmp/fake_ifa_tts.aiff"),
    )


def _mock_os_close_and_unlink():
    return (
        patch("ifa.services.tts_service.os.close"),
        patch("ifa.services.tts_service.os.unlink"),
    )


class TTSServiceDispatchTests(unittest.TestCase):
    def test_empty_text_is_noop(self):
        tts = TTSService()
        with patch("ifa.services.tts_service.subprocess.run") as run:
            tts.speak("")
            run.assert_not_called()

    def test_darwin_uses_say_to_file_then_afplay(self):
        tts = TTSService()
        close_patch, unlink_patch = _mock_os_close_and_unlink()
        with patch("ifa.services.tts_service.sys.platform", "darwin"), \
             _mock_tempfile(), close_patch, unlink_patch, \
             patch("ifa.services.tts_service.subprocess.run") as run:
            tts.speak("hello")
        self.assertEqual(run.call_count, 2)
        say_cmd = run.call_args_list[0][0][0]
        afplay_cmd = run.call_args_list[1][0][0]
        self.assertEqual(say_cmd[0], "say")
        self.assertEqual(say_cmd[1], "-o")
        self.assertEqual(say_cmd[3], "--")
        self.assertEqual(say_cmd[4], "hello")
        self.assertEqual(afplay_cmd[0], "afplay")
        self.assertEqual(say_cmd[2], afplay_cmd[1])

    def test_darwin_uses_argv_separator_for_text_starting_with_dash(self):
        """Text starting with `-` must not be interpreted as a `say` flag."""
        tts = TTSService()
        close_patch, unlink_patch = _mock_os_close_and_unlink()
        with patch("ifa.services.tts_service.sys.platform", "darwin"), \
             _mock_tempfile(), close_patch, unlink_patch, \
             patch("ifa.services.tts_service.subprocess.run") as run:
            tts.speak("--version")
        say_cmd = run.call_args_list[0][0][0]
        self.assertIn("--", say_cmd)
        self.assertEqual(say_cmd[-1], "--version")

    def test_windows_uses_encoded_command_with_env_var(self):
        tts = TTSService()
        with patch("ifa.services.tts_service.sys.platform", "win32"), \
             patch("ifa.services.tts_service.subprocess.run") as run:
            tts.speak("hello")
        run.assert_called_once()
        args, kwargs = run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "powershell")
        self.assertIn("-NoProfile", cmd)
        self.assertIn("-EncodedCommand", cmd)
        encoded = cmd[-1]
        decoded = base64.b64decode(encoded).decode("utf-16-le")
        self.assertIn("GetEnvironmentVariable", decoded)
        self.assertIn("IFA_TTS_TEXT", decoded)
        self.assertEqual(kwargs["env"]["IFA_TTS_TEXT"], "hello")

    def test_windows_injection_payload_does_not_appear_in_command(self):
        """Newlines and PowerShell metacharacters in text must not leak
        into the command string."""
        tts = TTSService()
        payload = "innocent\n; Start-Process calc"
        with patch("ifa.services.tts_service.sys.platform", "win32"), \
             patch("ifa.services.tts_service.subprocess.run") as run:
            tts.speak(payload)
        cmd = run.call_args[0][0]
        encoded = cmd[-1]
        decoded_script = base64.b64decode(encoded).decode("utf-16-le")
        self.assertNotIn("Start-Process calc", decoded_script)
        self.assertEqual(run.call_args[1]["env"]["IFA_TTS_TEXT"], payload)

    def test_linux_uses_espeak_with_argv_separator(self):
        tts = TTSService()
        with patch("ifa.services.tts_service.sys.platform", "linux"), \
             patch("ifa.services.tts_service.subprocess.run") as run:
            tts.speak("hello")
        run.assert_called_once()
        cmd = run.call_args[0][0]
        self.assertEqual(cmd, ["espeak", "--", "hello"])


class TTSServiceErrorHandlingTests(unittest.TestCase):
    def test_missing_binary_does_not_raise(self):
        tts = TTSService()
        err = FileNotFoundError("No such file or directory")
        err.filename = "say"
        with patch("ifa.services.tts_service.subprocess.run", side_effect=err), \
             _mock_tempfile(), \
             patch("ifa.services.tts_service.os.close"), \
             patch("ifa.services.tts_service.os.unlink"):
            tts.speak("hello")

    def test_generic_subprocess_error_does_not_raise(self):
        tts = TTSService()
        with patch("ifa.services.tts_service.subprocess.run",
                   side_effect=OSError("EPERM")), \
             _mock_tempfile(), \
             patch("ifa.services.tts_service.os.close"), \
             patch("ifa.services.tts_service.os.unlink"):
            tts.speak("hello")

    def test_darwin_cleanup_runs_even_on_subprocess_error(self):
        tts = TTSService()
        close_patch = patch("ifa.services.tts_service.os.close")
        unlink_patch = patch("ifa.services.tts_service.os.unlink")
        with patch("ifa.services.tts_service.sys.platform", "darwin"), \
             _mock_tempfile(), \
             close_patch, unlink_patch as unlink_mock, \
             patch("ifa.services.tts_service.subprocess.run",
                   side_effect=OSError("boom")):
            tts.speak("hello")
        unlink_mock.assert_called_once_with("/tmp/fake_ifa_tts.aiff")


class TTSServiceThreadSafetyTests(unittest.TestCase):
    def test_concurrent_speak_calls_from_threads(self):
        tts = TTSService()
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd[-1])

        with patch("ifa.services.tts_service.subprocess.run", side_effect=fake_run), \
             patch("ifa.services.tts_service.sys.platform", "linux"):
            threads = [
                threading.Thread(target=lambda i=i: tts.speak(f"from_thread_{i}"))
                for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=1.0)
                self.assertFalse(t.is_alive())

        self.assertEqual(sorted(calls), [f"from_thread_{i}" for i in range(5)])


class ManagerDecouplingTests(unittest.TestCase):
    """Plan R14: manager.py must not construct its own TTSService."""

    def test_manager_does_not_instantiate_tts_service(self):
        manager_path = pathlib.Path(__file__).parent.parent / "skills" / "manager.py"
        tree = ast.parse(manager_path.read_text())

        bad_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "TTSService":
                    bad_calls.append(node.lineno)
                elif isinstance(func, ast.Attribute) and func.attr == "TTSService":
                    bad_calls.append(node.lineno)

        self.assertEqual(
            bad_calls,
            [],
            f"manager.py calls TTSService() at lines {bad_calls} — should receive it via handle_with_intent parameter",
        )


if __name__ == "__main__":
    unittest.main()
