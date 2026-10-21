from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from launcher.agent_process import AgentProcess
from launcher.system_info import check_ollama, nvidia_gpus, venv_python

TORCH_VERSION = "2.6.0"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
PYTHON = (3, 12)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    help: str
    argv: list[str] = field(default_factory=list)
    done: bool | None = None
    blocked: str | None = None
    detached: bool = False
    rerun: bool = True
    env: dict[str, str] = field(default_factory=dict)


def _site_packages(repo_root: Path) -> list[Path]:
    venv = repo_root / "venv"
    return [venv / "Lib" / "site-packages", *sorted((venv / "lib").glob("python*/site-packages"))]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def installed_versions(repo_root: Path) -> dict[str, str]:
    found = {}
    for site in _site_packages(repo_root):
        if not site.is_dir():
            continue
        for entry in site.iterdir():
            if entry.name.endswith(".dist-info"):
                name, _, version = entry.name[: -len(".dist-info")].rpartition("-")
                found.setdefault(_norm(name), version)
    return found


def installed_version(repo_root: Path, dist: str) -> str | None:
    return installed_versions(repo_root).get(_norm(dist))


def venv_python_version(repo_root: Path) -> tuple[int, int] | None:
    try:
        text = (repo_root / "venv" / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^version(?:_info)?\s*=\s*(\d+)\.(\d+)", text, re.MULTILINE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def wrong_versions(repo_root: Path, constraints: Path) -> list[tuple[str, str, str]]:
    try:
        lines = constraints.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    installed = installed_versions(repo_root)
    wrong = []
    for line in lines:
        name, sep, wanted = line.split("#", 1)[0].strip().partition("==")
        have = installed.get(_norm(name)) if sep else None
        if have is not None and have.split("+", 1)[0] != wanted.strip():
            wrong.append((name, have, wanted.strip()))
    return wrong


def _marker_applies(marker: str) -> bool:
    try:
        from packaging.markers import Marker
        return Marker(marker).evaluate()
    except ImportError:
        return sys.platform in marker or "sys_platform" not in marker
    except Exception:
        return True


def missing_requirements(repo_root: Path, requirements: Path) -> list[str]:
    try:
        lines = requirements.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    installed = installed_versions(repo_root)
    missing = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        spec, _, marker = line.partition(";")
        if marker.strip() and not _marker_applies(marker.strip()):
            continue
        name = re.split(r"[<>=!~\[\s]", spec.strip(), maxsplit=1)[0]
        if name and _norm(name) not in installed:
            missing.append(name)
    return missing


def display(argv: list[str], repo_root: Path) -> str:
    root = str(repo_root) + os.sep
    parts = [a[len(root):] if a.startswith(root) else a for a in argv]
    return subprocess.list2cmdline(parts) if sys.platform == "win32" else shlex.join(parts)


def _ollama_host(url: str) -> tuple[str, bool]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    return f"{host}:{parsed.port or 11434}", host in LOCAL_HOSTS


def _install_ollama() -> tuple[list[str], str | None]:
    if sys.platform == "darwin":
        return (["brew", "install", "ollama"], None) if shutil.which("brew") else (
            [], "Install Homebrew, or download Ollama from https://ollama.com/download")
    if sys.platform == "win32":
        return (["winget", "install", "--id", "Ollama.Ollama", "-e"], None) if shutil.which("winget") else (
            [], "Download Ollama from https://ollama.com/download")
    return [], "Install Ollama with your package manager or from https://ollama.com/download"


def _enrolled(repo_root: Path, source: str) -> bool:
    import json

    try:
        profile = json.loads((repo_root / "voiceprint.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool((profile.get("sources") or {}).get(source, {}).get("embeddings"))


def plan(repo_root: Path, values: dict[str, str], default_url: str) -> list[Step]:
    py = venv_python(repo_root)
    vpy = str(py) if py else str(repo_root / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
    no_venv = None if py else "Create the virtual environment first"
    pip = [vpy, "-m", "pip", "install"]
    requirements = repo_root / "ifa" / "requirements.txt"
    constraints = repo_root / "ifa" / "constraints.txt"
    missing = missing_requirements(repo_root, requirements) if py else []
    wrong = wrong_versions(repo_root, constraints) if py else []
    venv_version = venv_python_version(repo_root) if py else None
    bad_python = (
        f"Ifa needs Python {PYTHON[0]}.{PYTHON[1]} but the venv has {venv_version[0]}.{venv_version[1]}. "
        f"Install Python {PYTHON[0]}.{PYTHON[1]}, delete the venv folder, and run Setup again"
        if venv_version and venv_version != PYTHON else None
    )
    problems = []
    if missing:
        problems.append(f"Missing: {', '.join(missing[:6])}{' and more' if len(missing) > 6 else ''}.")
    if wrong:
        shown = ", ".join(f"{n} {have} (needs {want})" for n, have, want in wrong[:4])
        problems.append(f"Wrong versions: {shown}{' and more' if len(wrong) > 4 else ''}. Running this fixes them.")

    url = values.get("MAIN_OLLAMA_URL") or default_url
    host, local = _ollama_host(url)
    model = values.get("IFA_OLLAMA_MODEL", "").strip()
    has_ollama = shutil.which("ollama") is not None
    ollama = check_ollama(url, model, timeout=1.0)

    steps = [
        Step("venv", "Create the virtual environment",
             f"A private Python {PYTHON[0]}.{PYTHON[1]} for Ifa in ./venv, so its packages don't touch the system Python.",
             [sys.executable, "-m", "venv", str(repo_root / "venv")], done=py is not None, rerun=False,
             blocked=None if py or sys.version_info[:2] == PYTHON else
             f"This launcher runs on Python {sys.version_info[0]}.{sys.version_info[1]}; Ifa needs {PYTHON[0]}.{PYTHON[1]}. "
             f"Install Python {PYTHON[0]}.{PYTHON[1]} and open the launcher with it"),
        Step("requirements", "Install packages",
             "Everything Ifa uses (speech-to-text, the Chatterbox voice with PyTorch, the voice check, skills) "
             "at the exact versions in ifa/constraints.txt, which are tested together. A few GB the first time."
             + (" " + " ".join(problems) if py and problems else ""),
             [*pip, "-r", str(requirements), "-c", str(constraints)],
             done=py is not None and not missing and not wrong and not bad_python,
             blocked=no_venv or bad_python),
    ]
    if sys.platform != "darwin" and nvidia_gpus():
        torch = installed_version(repo_root, "torch") or ""
        steps.append(Step(
            "torch_cuda", "Switch PyTorch to the CUDA build",
            "PyPI ships CPU-only PyTorch on Windows. This swaps in the GPU build of the same version Chatterbox needs.",
            [*pip, "--force-reinstall", "--no-deps", f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}",
             "--index-url", TORCH_CUDA_INDEX],
            done="+cu" in torch, blocked=no_venv or (None if torch else "Install packages first")))
    install_argv, install_blocked = _install_ollama()
    steps += [
        Step("ollama_install", "Install Ollama", "The local server that runs the language model.",
             install_argv, done=has_ollama, blocked=None if has_ollama else install_blocked, rerun=False),
        Step("ollama_start", "Start Ollama", f"Runs ollama serve in the background on {host}.",
             ["ollama", "serve"], done=ollama["ok"], detached=True, env={"OLLAMA_HOST": host}, rerun=False,
             blocked=f"{url} is another machine. Start Ollama there" if not local
             else None if has_ollama else "Install Ollama first"),
        Step("ollama_pull", "Download the model", f"Pulls {model or 'the configured model'} into Ollama (one time).",
             ["ollama", "pull", model or "<model>"], done=bool(ollama.get("model_present")), env={"OLLAMA_HOST": host},
             blocked="Set the model under Language model" if not model else "Install Ollama first" if not has_ollama
             else None if ollama["ok"] else "Start Ollama first"),
        Step("spotify_login", "Log in to Spotify",
             "Opens Spotify in your browser once and saves the login to .spotify_token, so Ifa can control playback.",
             [vpy, "-m", "ifa.skills.spotify"], done=(repo_root / ".spotify_token").exists(),
             blocked=no_venv or ("Add your Spotify keys under Keys first"
                                 if not (values.get("SPOTIFY_CLIENT_ID") and values.get("SPOTIFY_CLIENT_SECRET"))
                                 else None if installed_version(repo_root, "spotipy")
                                 else "Install packages first")),
        Step("voice_models", "Download voice models",
             "Caches the wake-word and Whisper models so voice mode works offline. Safe to re-run.",
             [vpy, "-m", "scripts.setup_voice_models"], blocked=no_venv),
        Step("voice_enroll", "Teach Ifa your voice (PC mic)",
             "Read 6 short phrases aloud so Ifa ignores other voices. Re-run to record again. "
             "The phone app has the same thing under Settings.",
             [vpy, "-m", "ifa.voice.enroll", "--source", "pc"], done=_enrolled(repo_root, "pc"),
             blocked=no_venv or (None if installed_version(repo_root, "sherpa_onnx") else "Install packages first")),
    ]
    return steps


class SetupRunner:
    def __init__(self, repo_root: Path, runner: AgentProcess, plan_fn) -> None:
        self.repo_root = repo_root
        self.runner = runner
        self._plan = plan_fn
        self._lock = threading.Lock()
        self._current: str | None = None
        self._cancel = threading.Event()

    def status(self) -> dict:
        with self._lock:
            return {"running": self._current is not None, "step": self._current}

    def run(self, ids: list[str], only_missing: bool) -> None:
        known = {s.id for s in self._plan()}
        unknown = [i for i in ids if i not in known]
        if unknown or not ids:
            raise ValueError(f"unknown setup step: {', '.join(unknown) or '(none)'}")
        with self._lock:
            if self._current is not None:
                raise RuntimeError("setup is already running")
            self._current = ids[0]
        self._cancel.clear()
        threading.Thread(target=self._run, args=(ids, only_missing), daemon=True).start()

    def stop(self) -> None:
        self._cancel.set()
        self.runner.stop()

    def _run(self, ids: list[str], only_missing: bool) -> None:
        log = self.runner.log.append
        try:
            for step_id in ids:
                if self._cancel.is_set():
                    log("[launcher] setup cancelled")
                    return
                step = next(s for s in self._plan() if s.id == step_id)
                with self._lock:
                    self._current = step.id
                if only_missing and step.done:
                    continue
                if step.blocked:
                    log(f"[launcher] setup: {step.label}: {step.blocked}")
                    return
                log(f"[launcher] setup: {step.label}")
                log(f"[launcher] $ {display(step.argv, self.repo_root)}")
                if step.detached:
                    self._spawn_detached(step)
                    continue
                try:
                    self.runner.start(argv=step.argv, extra_env=step.env)
                except OSError as exc:
                    log(f"[launcher] could not run {step.argv[0]}: {exc}")
                    return
                code = self.runner.wait()
                if code != 0:
                    log(f"[launcher] setup stopped: {step.label} failed (exit code {code})")
                    return
            log("[launcher] setup finished")
        finally:
            with self._lock:
                self._current = None

    def _spawn_detached(self, step: Step) -> None:
        kwargs: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS} \
            if sys.platform == "win32" else {"start_new_session": True}
        try:
            subprocess.Popen(step.argv, env={**os.environ, **step.env}, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
        except OSError as exc:
            self.runner.log.append(f"[launcher] could not start {step.argv[0]}: {exc}")
            return
        for _ in range(30):
            if self._cancel.wait(0.5):
                return
            if next(s for s in self._plan() if s.id == step.id).done:
                self.runner.log.append("[launcher] Ollama is up")
                return
        self.runner.log.append("[launcher] Ollama didn't answer within 15 seconds")
