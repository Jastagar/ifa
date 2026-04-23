"""Pre-download the voice-mode model weights.

Run once as part of `run.bat` / `run.ps1` / `run.command` so the
runtime launch can set ``HF_HUB_OFFLINE=1`` and never touch the
network afterward.

Models fetched:
  - openWakeWord ``hey_mycroft`` (~1 MB) — wake-word detector
  - faster-whisper ``small.en`` (~470 MB) — STT

Both are idempotent: if the model is already cached locally, the call
returns immediately without hitting the network. The script is safe
to re-run on every launch.

Run:
    PYTHONPATH=. python -m scripts.setup_voice_models
"""
from __future__ import annotations

import os
import sys


def _step(msg: str) -> None:
    print(f"[setup-voice] {msg}", flush=True)


def ensure_openwakeword_models() -> None:
    model_name = os.environ.get("IFA_WAKE_MODEL", "hey_mycroft")
    # Skip download if this is a custom .onnx path — those don't live in the cache.
    if os.path.exists(model_name):
        _step(f"wake-word model is a custom path: {model_name}  (no download)")
        return

    _step(f"ensuring openWakeWord model '{model_name}' is cached...")
    from openwakeword.utils import download_models
    download_models(model_names=[model_name])
    _step("openWakeWord ready.")


def ensure_whisper_model() -> None:
    model_name = os.environ.get("IFA_WHISPER_MODEL", "small.en")
    _step(f"ensuring faster-whisper model '{model_name}' is cached...")
    # Constructing the model triggers a download to the HF cache if missing.
    # On subsequent runs, this hits the cache and is fast.
    from faster_whisper import WhisperModel
    WhisperModel(model_name, device="cpu", compute_type="int8")
    _step("faster-whisper ready.")


def main() -> int:
    try:
        ensure_openwakeword_models()
        ensure_whisper_model()
    except Exception as exc:
        _step(f"ERROR: {exc}")
        return 1
    _step("all voice models cached. runtime can now use HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
