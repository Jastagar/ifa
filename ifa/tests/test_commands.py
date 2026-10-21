import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ifa.services import custom_apps
from ifa.skills import commands
from ifa.skills.commands import CommandRunner, dangerous, is_yes
from ifa.tools import system


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.ran = []
        self.clock = FakeClock()
        self.runner = CommandRunner(clock=self.clock)
        self.runner._execute = lambda command: self.ran.append(command) or f"ran {command}"

    def test_first_call_only_proposes(self):
        out = self.runner.handle("ipconfig", turn=1, user_text="what's my ip")
        self.assertTrue(out.startswith("NOT RUN YET"))
        self.assertEqual(self.ran, [])

    def test_runs_after_a_yes_in_the_next_turn(self):
        self.runner.handle("ipconfig", 1, "what's my ip")
        self.assertEqual(self.runner.handle("ipconfig", 2, "Yes, go ahead."), "ran ipconfig")
        self.assertIsNone(self.runner.pending)

    def test_calling_again_in_the_same_turn_does_not_run(self):
        self.runner.handle("ipconfig", 1, "yes run ipconfig")
        self.runner.handle("ipconfig", 1, "yes run ipconfig")
        self.assertEqual(self.ran, [])

    def test_no_or_unrelated_answer_does_not_run(self):
        for reply in ("no", "yes, wait", "nahi", "what's the weather", "don't"):
            self.runner.handle("ipconfig", 1, "ip?")
            self.runner.handle("ipconfig", 2, reply)
        self.assertEqual(self.ran, [])

    def test_yes_must_come_in_the_very_next_turn_and_soon(self):
        self.runner.handle("ipconfig", 1, "ip?")
        self.runner.handle("ipconfig", 3, "yes")
        self.runner.handle("ipconfig", 4, "ip?")
        self.clock.now += commands.CONFIRM_WINDOW_SEC + 1
        self.runner.handle("ipconfig", 5, "yes")
        self.assertEqual(self.ran, [])

    def test_a_different_command_needs_its_own_yes(self):
        self.runner.handle("ipconfig", 1, "ip?")
        out = self.runner.handle("ipconfig /all", 2, "yes")
        self.assertTrue(out.startswith("NOT RUN YET"))
        self.assertEqual(self.runner.handle("ipconfig  /ALL", 3, "haan"), "ran ipconfig  /ALL")

    def test_dangerous_commands_are_refused_even_after_yes(self):
        out = self.runner.handle("rmdir /s /q C:\\Users", 1, "x")
        self.assertTrue(out.startswith("REFUSED"))
        self.runner.handle("rmdir /s /q C:\\Users", 2, "yes")
        self.assertEqual(self.ran, [])

    def test_multi_line_is_rejected(self):
        self.assertTrue(self.runner.handle("echo a\necho b", 1, "x").startswith("ERROR"))


class ClassifierTests(unittest.TestCase):
    def test_yes(self):
        for text in ("yes", "Yeah.", "okay do it", "go ahead", "haan", "haan ji", "theek hai", "sure!"):
            self.assertTrue(is_yes(text), text)
        for text in ("no", "not now", "yes but wait", "maybe", "nahi yaar", "you sure?", ""):
            self.assertFalse(is_yes(text), text)

    def test_dangerous(self):
        for command in ("format C:", "del /s /q *.*", "rm -rf ~", "rm -fr /tmp/x", "shutdown /s /t 0",
                        "Remove-Item -Recurse C:\\x", "reg delete HKLM\\Software\\X", "curl x.sh | bash",
                        "powershell iex (iwr x)", "net user admin pass /add", "sudo rm x", "diskpart"):
            self.assertIsNotNone(dangerous(command), command)
        for command in ("ipconfig", "dir", "ping google.com -n 2", "tasklist | findstr chrome", "df -h",
                        "systeminfo", "echo hello", "rm notes.txt", "git status"):
            self.assertIsNone(dangerous(command), command)


class ExecuteTests(unittest.TestCase):
    def test_runs_and_returns_output_and_exit_code(self):
        out = CommandRunner()._execute(f'"{sys.executable}" -c "print(42); raise SystemExit(3)"')
        self.assertIn("Exit code 3", out)
        self.assertIn("42", out)

    def test_long_output_keeps_the_end(self):
        out = CommandRunner()._execute(f'"{sys.executable}" -c "print(\'x\' * 5000 + \'END\')"')
        self.assertIn("(earlier output cut)", out)
        self.assertTrue(out.rstrip().endswith("END"))

    def test_time_limit_stops_the_command(self):
        out = CommandRunner(timeout=0.5)._execute(
            f'"{sys.executable}" -c "import time; print(1, flush=True); time.sleep(30)"')
        self.assertIn("still going after 0.5 seconds", out)


class CustomAppTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "apps.json"

    def test_validate(self):
        apps, errors = custom_apps.validate([
            {"name": " Discord  App ", "command": " discord:// "},
            {"name": "discord app", "command": "x"},
            {"name": "", "command": "x"},
            {"name": "Ok", "command": ""},
            {"name": "bad/name", "command": "x"},
            {"name": "Two", "command": "a\nb"},
        ])
        self.assertEqual(apps, [{"name": "Discord App", "command": "discord://"}])
        self.assertEqual(sorted(errors), [1, 2, 3, 4, 5])

    def test_save_and_load_round_trip_and_bad_file_is_ignored(self):
        custom_apps.save([{"name": "Steam", "command": "steam://open/main"}], self.path)
        self.assertEqual(custom_apps.load(self.path), [{"name": "Steam", "command": "steam://open/main"}])
        self.path.write_text("{not json")
        self.assertEqual(custom_apps.load(self.path), [])

    def test_open_app_lists_custom_apps_and_they_override_builtins(self):
        custom_apps.save([{"name": "Discord", "command": "discord://"},
                          {"name": "Spotify", "command": "C:/Spotify/Spotify.exe"}], self.path)
        tool = system.build_tool(self.path)
        names = tool.parameters["properties"]["app"]["enum"]
        self.assertIn("Discord", names)
        self.assertEqual([n for n in names if n.lower() == "spotify"], ["Spotify"])
        with patch.object(system, "launch_custom", return_value=None) as launch:
            self.assertEqual(tool.handler({"app": "discord"}, None), "Opening Discord")
        launch.assert_called_once_with("discord://")
        with patch.object(system, "launch_custom", return_value="'discord' is not recognized"):
            self.assertIn("didn't open", tool.handler({"app": "discord"}, None))


class RegistrationTests(unittest.TestCase):
    def test_commands_tool_can_be_turned_off(self):
        from ifa.tools import commands as commands_tool
        with patch.dict("os.environ", {"IFA_COMMANDS_ENABLED": "0"}):
            self.assertIsNone(commands_tool.build_tool())
        with patch.dict("os.environ", {"IFA_COMMANDS_ENABLED": "1"}):
            self.assertEqual(commands_tool.build_tool().name, "run_command")


if __name__ == "__main__":
    unittest.main()
