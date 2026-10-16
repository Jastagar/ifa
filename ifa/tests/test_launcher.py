import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from launcher import env_file
from launcher.agent_process import AgentProcess
from launcher.server import Launcher, SettingsError, serve
from launcher.settings_schema import BY_KEY, SETTINGS, normalize

from dotenv import dotenv_values


class EnvFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / ".env"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def test_updates_in_place_and_keeps_comments_and_unknown_keys(self):
        self.write("# my notes\nIFA_MODE=text\n\nOTHER=keep me\nIFA_OLLAMA_MODEL=old\n")
        env_file.update_env(self.path, {"IFA_MODE": "voice", "IFA_OLLAMA_MODEL": None, "NEW_ONE": "x"})
        self.assertEqual(
            self.path.read_text(),
            "# my notes\nIFA_MODE=voice\n\nOTHER=keep me\n\nNEW_ONE=x\n",
        )

    def test_duplicate_keys_collapse_to_one(self):
        self.write("A=1\nA=2\n")
        env_file.update_env(self.path, {"A": "3"})
        self.assertEqual(self.path.read_text(), "A=3\n")

    def test_values_round_trip_through_python_dotenv(self):
        tricky = {
            "WIN_PATH": r"c:\temp\ifa_dump",
            "SPACES": "hello world",
            "HASH": "abc#def",
            "QUOTES": "it's \"quoted\"",
            "URL": "http://127.0.0.1:8888/callback",
        }
        env_file.update_env(self.path, tricky)
        self.assertEqual(dict(dotenv_values(self.path)), tricky)
        self.assertEqual(env_file.read_env(self.path), tricky)

    def test_read_handles_export_quotes_and_inline_comments(self):
        self.write("export A=1\nB='two words'\nC=\"x\" \nD=value # note\n# E=hidden\n")
        self.assertEqual(env_file.read_env(self.path), {"A": "1", "B": "two words", "C": "x", "D": "value"})

    def test_creates_missing_file(self):
        env_file.update_env(self.path, {"A": "1"})
        self.assertEqual(self.path.read_text(), "A=1\n")


class NormalizeTests(unittest.TestCase):
    def test_bool_accepts_loose_spellings_and_writes_1_or_0(self):
        s = BY_KEY["IFA_OLLAMA_THINK"]
        self.assertEqual(normalize(s, "True"), "1")
        self.assertEqual(normalize(s, "False,"), "0")

    def test_number_range_and_format(self):
        s = BY_KEY["IFA_WAKE_THRESHOLD"]
        self.assertEqual(normalize(s, " 0.40 "), "0.4")
        self.assertEqual(normalize(BY_KEY["IFA_VAD_SILENCE_MS"], "900.0"), "900")
        for bad in ("abc", "1.5", "-1"):
            with self.assertRaises(ValueError):
                normalize(s, bad)

    def test_select_must_be_an_option(self):
        with self.assertRaises(ValueError):
            normalize(BY_KEY["IFA_TTS_DEVICE"], "tpu")
        self.assertEqual(normalize(BY_KEY["IFA_TTS_DEVICE"], "cpu"), "cpu")

    def test_no_newlines(self):
        with self.assertRaises(ValueError):
            normalize(BY_KEY["IFA_OLLAMA_MODEL"], "a\nIFA_MODE=voice")

    def test_every_default_is_valid(self):
        for s in SETTINGS:
            if s.default:
                normalize(s, s.default)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        (self.root / ".env").write_text("IFA_MODE=text\nSPOTIFY_CLIENT_SECRET=s3cret\nCLAUDE_KEY=abc\n")
        self.launcher = Launcher(self.root, agent=AgentProcess([sys.executable, "-c", "pass"], cwd=str(self.root)))

    def field(self, data, key):
        return next(f for g in data["groups"] for f in g["settings"] if f["key"] == key)

    def test_settings_never_include_secret_values(self):
        data = self.launcher.settings()
        secret = self.field(data, "SPOTIFY_CLIENT_SECRET")
        self.assertTrue(secret["is_set"])
        self.assertNotIn("value", secret)
        self.assertNotIn("s3cret", json.dumps(data))
        self.assertNotIn("abc", json.dumps(data))
        self.assertEqual(data["custom_keys"], ["CLAUDE_KEY"])

    def test_secret_null_keeps_empty_clears_text_replaces(self):
        self.launcher.save({"values": {"SPOTIFY_CLIENT_SECRET": None}})
        self.assertEqual(env_file.read_env(self.root / ".env")["SPOTIFY_CLIENT_SECRET"], "s3cret")
        self.launcher.save({"values": {"SPOTIFY_CLIENT_SECRET": "new"}})
        self.assertEqual(env_file.read_env(self.root / ".env")["SPOTIFY_CLIENT_SECRET"], "new")
        self.launcher.save({"values": {"SPOTIFY_CLIENT_SECRET": ""}})
        self.assertNotIn("SPOTIFY_CLIENT_SECRET", env_file.read_env(self.root / ".env"))

    def test_empty_value_removes_the_key_so_the_default_applies(self):
        self.launcher.save({"values": {"IFA_MODE": ""}})
        self.assertNotIn("IFA_MODE", env_file.read_env(self.root / ".env"))

    def test_invalid_values_save_nothing(self):
        with self.assertRaises(SettingsError) as cm:
            self.launcher.save({"values": {"IFA_MODE": "voice", "IFA_WAKE_THRESHOLD": "7", "NOPE": "1"}})
        self.assertEqual(set(cm.exception.errors), {"IFA_WAKE_THRESHOLD", "NOPE"})
        self.assertEqual(env_file.read_env(self.root / ".env")["IFA_MODE"], "text")

    def test_custom_variables(self):
        self.launcher.save({"custom": {"MY_KEY": "v", "CLAUDE_KEY": None}})
        values = env_file.read_env(self.root / ".env")
        self.assertEqual(values["MY_KEY"], "v")
        self.assertNotIn("CLAUDE_KEY", values)
        for bad in ("lower", "IFA_MODE", "1BAD"):
            with self.assertRaises(SettingsError):
                self.launcher.save({"custom": {bad: "x"}})


class ServerSecurityTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        launcher = Launcher(Path(self.dir.name), agent=AgentProcess([sys.executable, "-c", "pass"], cwd=self.dir.name))
        self.server, self.token = serve(launcher, port=0, token="t0ken")
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, token="t0ken", host=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Host": host or f"127.0.0.1:{self.port}"}
        if token is not None:
            headers["X-Launcher-Token"] = token
        if body is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read().decode()

    def test_api_requires_the_token(self):
        self.assertEqual(self.request("GET", "/api/settings")[0], 200)
        self.assertEqual(self.request("GET", "/api/settings", token=None)[0], 403)
        self.assertEqual(self.request("POST", "/api/settings", token="wrong", body={})[0], 403)

    def test_foreign_host_header_is_rejected(self):
        self.assertEqual(self.request("GET", "/", host="evil.example:80")[0], 403)
        self.assertEqual(self.request("GET", "/api/settings", host="evil.example")[0], 403)

    def test_page_embeds_the_token(self):
        status, html = self.request("GET", "/", token=None)
        self.assertEqual(status, 200)
        self.assertIn('const TOKEN = "t0ken"', html)

    def test_bad_settings_return_field_errors(self):
        status, body = self.request("POST", "/api/settings", body={"values": {"IFA_TTS_DEVICE": "tpu"}})
        self.assertEqual(status, 400)
        self.assertIn("IFA_TTS_DEVICE", json.loads(body)["fields"])


class AgentProcessTests(unittest.TestCase):
    ECHO = "import sys\nprint('ready', flush=True)\nfor line in sys.stdin:\n    print('got:' + line.strip(), flush=True)\n"

    def wait_for(self, agent, text, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(text in line["text"] for line in agent.logs_after(0)[0]):
                return True
            time.sleep(0.05)
        return False

    def test_start_send_stop(self):
        agent = AgentProcess([sys.executable, "-c", self.ECHO], cwd=tempfile.gettempdir())
        agent.start()
        self.addCleanup(agent.stop)
        self.assertEqual(agent.status()["state"], "running")
        self.assertTrue(self.wait_for(agent, "ready"))
        agent.send("hello")
        self.assertTrue(self.wait_for(agent, "got:hello"))
        with self.assertRaises(RuntimeError):
            agent.start()
        agent.stop()
        self.assertEqual(agent.status()["state"], "exited")
        with self.assertRaises(RuntimeError):
            agent.send("late")

    def test_managed_variables_are_not_inherited(self):
        import os
        os.environ["IFA_LAUNCHER_TEST_VAR"] = "from-shell"
        self.addCleanup(os.environ.pop, "IFA_LAUNCHER_TEST_VAR", None)
        script = "import os; print('var=' + os.environ.get('IFA_LAUNCHER_TEST_VAR', 'unset'), flush=True)"
        agent = AgentProcess([sys.executable, "-c", script], cwd=tempfile.gettempdir())
        agent.start(clear_env={"IFA_LAUNCHER_TEST_VAR"})
        self.addCleanup(agent.stop)
        self.assertTrue(self.wait_for(agent, "var=unset"))

    def test_logs_strip_ansi_colors(self):
        agent = AgentProcess([sys.executable, "-c", "print('\\x1b[31mred\\x1b[0m', flush=True)"], cwd=tempfile.gettempdir())
        agent.start()
        self.addCleanup(agent.stop)
        self.assertTrue(self.wait_for(agent, "red"))
        self.assertFalse(any("\x1b" in line["text"] for line in agent.logs_after(0)[0]))


if __name__ == "__main__":
    unittest.main()
