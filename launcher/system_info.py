from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def venv_python(repo_root: Path) -> Path | None:
    candidates = [repo_root / "venv" / "Scripts" / "python.exe", repo_root / "venv" / "bin" / "python"]
    return next((p for p in candidates if p.exists()), None)


def _nvidia_gpus() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def detect_system(repo_root: Path) -> dict:
    system = platform.system()
    gpus = _nvidia_gpus()
    apple_silicon = system == "Darwin" and platform.machine() == "arm64"
    python = venv_python(repo_root)
    return {
        "os": {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(system, system),
        "os_version": platform.release() if system != "Darwin" else platform.mac_ver()[0],
        "arch": platform.machine(),
        "cuda_gpus": gpus,
        "apple_silicon": apple_silicon,
        "recommended_device": "cuda" if gpus else "cpu",
        "launcher_python": sys.version.split()[0],
        "agent_python": str(python.relative_to(repo_root)) if python else None,
        "torch_in_launcher": importlib.util.find_spec("torch") is not None,
    }


def check_ollama(url: str, model: str, timeout: float = 2.0) -> dict:
    base = (url or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            models = [m["name"] for m in json.load(resp).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "url": base, "error": str(getattr(exc, "reason", exc)), "models": []}
    wanted = model.strip()
    present = any(m == wanted or m == f"{wanted}:latest" for m in models) if wanted else False
    return {"ok": True, "url": base, "models": models, "model_present": present}
