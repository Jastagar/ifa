import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ifa.utils import offline


def fake_repo(cache: Path, repo_id: str, files=("config.json", "model.safetensors")) -> None:
    snapshot = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    for name in files:
        (snapshot / name).write_text("x")


class OfflineModeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.cache = Path(self.dir.name)
        env = patch.dict(os.environ, {"IFA_WHISPER_MODEL": "org/whisper"}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for var in ("IFA_OFFLINE", *offline.OFFLINE_VARS):
            os.environ.pop(var, None)

    def test_auto_goes_offline_once_every_model_is_downloaded(self):
        fake_repo(self.cache, "org/whisper")
        fake_repo(self.cache, offline.CHATTERBOX_REPO)
        status = offline.configure(self.cache)
        self.assertTrue(status.startswith("offline"))
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_auto_stays_online_and_names_what_is_missing(self):
        fake_repo(self.cache, "org/whisper")
        status = offline.configure(self.cache)
        self.assertIn(offline.CHATTERBOX_REPO, status)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")

    def test_auto_overrides_a_stale_hf_hub_offline_0(self):
        os.environ["HF_HUB_OFFLINE"] = "0"
        fake_repo(self.cache, "org/whisper")
        fake_repo(self.cache, offline.CHATTERBOX_REPO)
        offline.configure(self.cache)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_a_half_downloaded_repo_does_not_count(self):
        fake_repo(self.cache, "org/whisper", files=("config.json",))
        fake_repo(self.cache, offline.CHATTERBOX_REPO)
        self.assertFalse(offline.is_cached("org/whisper", self.cache))
        self.assertTrue(offline.configure(self.cache).startswith("online"))

    def test_explicit_on_and_off(self):
        os.environ["IFA_OFFLINE"] = "on"
        self.assertTrue(offline.configure(self.cache).startswith("offline"))
        os.environ["IFA_OFFLINE"] = "off"
        fake_repo(self.cache, "org/whisper")
        fake_repo(self.cache, offline.CHATTERBOX_REPO)
        self.assertTrue(offline.configure(self.cache).startswith("online"))
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")

    def test_a_local_model_folder_counts_as_downloaded(self):
        self.assertTrue(offline.is_cached(self.dir.name, self.cache))


class NoInternetAtStartupTests(unittest.TestCase):
    def test_loading_the_whatsapp_tool_does_not_import_pywhatkit(self):
        sys.modules.pop("pywhatkit", None)
        sys.modules.pop("pywhatkit.whats", None)
        from ifa.tools import whatsApp
        importlib.reload(whatsApp)
        self.assertNotIn("pywhatkit", sys.modules)


class RequirementsCheckTests(unittest.TestCase):
    def test_lists_missing_packages_and_skips_other_platforms(self):
        from launcher.setup_steps import missing_requirements

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            site = root / "venv" / "lib" / "python3.12" / "site-packages"
            site.mkdir(parents=True)
            (site / "numpy-2.2.6.dist-info").mkdir()
            (site / "python_dotenv-1.1.1.dist-info").mkdir()
            req = root / "requirements.txt"
            req.write_text(
                "# comment\nnumpy==2.2.6\npython-dotenv==1.1.1\nsherpa-onnx==1.13.8\n"
                'pywin32==311; sys_platform == "nonexistent-os"\nscipy>=1.10,<2.0\n'
            )
            self.assertEqual(missing_requirements(root, req), ["sherpa-onnx", "scipy"])


class LockFileTests(unittest.TestCase):
    def test_finds_drifted_versions_but_accepts_the_cuda_build_of_torch(self):
        from launcher.setup_steps import wrong_versions

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            site = root / "venv" / "Lib" / "site-packages"
            site.mkdir(parents=True)
            for dist in ("numpy-2.2.6", "sympy-1.14.0", "torch-2.6.0+cu124", "transformers-5.2.0", "PyYAML-6.0.3"):
                (site / f"{dist}.dist-info").mkdir()
            lock = root / "constraints.txt"
            lock.write_text("# lock\nnumpy==1.26.4\nsympy==1.13.1\ntorch==2.6.0\ntransformers==5.2.0\npyyaml==6.0.3\nscipy==1.16.0\n")
            self.assertEqual(wrong_versions(root, lock), [("numpy", "2.2.6", "1.26.4"), ("sympy", "1.14.0", "1.13.1")])

    def test_every_exact_pin_in_requirements_agrees_with_the_lock(self):
        import re

        root = Path(__file__).resolve().parents[1]
        lock = {}
        for line in (root / "constraints.txt").read_text().splitlines():
            name, sep, version = line.partition("==")
            if sep:
                lock[re.sub(r"[-_.]+", "-", name).lower()] = version
        for line in (root / "requirements.txt").read_text().splitlines():
            match = re.match(r"([A-Za-z0-9_.\-]+)\s*==\s*([^\s,;]+)", line.split("#")[0])
            if match and re.sub(r"[-_.]+", "-", match.group(1)).lower() in lock:
                self.assertEqual(lock[re.sub(r"[-_.]+", "-", match.group(1)).lower()], match.group(2), match.group(1))


class SpeakerStatusTests(unittest.TestCase):
    def test_missing_sherpa_turns_the_check_off_with_a_reason(self):
        from ifa.voice.speaker import SpeakerCheck

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "voiceprint.json"
            path.write_text('{"model": "%s", "sources": {"pc": {"embeddings": [[1.0, 0.0]]}}}'
                            % __import__("ifa.voice.speaker", fromlist=["MODEL_NAME"]).MODEL_NAME)
            check = SpeakerCheck(profile_path=path, model_path=Path(d) / "model.onnx")
            with patch("importlib.util.find_spec", return_value=None):
                self.assertFalse(check.active)
                self.assertIn("sherpa-onnx is not installed", check.status())
            self.assertIn("speaker model isn't downloaded", check.status())


if __name__ == "__main__":
    unittest.main()
