"""Pre-download the voice-mode model weights.

Run once as part of `run.bat` / `run.ps1` / `run.command` so the
runtime launch can set ``HF_HUB_OFFLINE=1`` and never touch the
network afterward.

Models fetched (the wake-word and Whisper variants come from the
.env / shell env, so this script sees the same models the runtime
will eventually load):
  - openWakeWord (e.g. ``ifa``) — wake-word detector
  - HuggingFace Whisper (e.g. ``Oriserve/Whisper-Hindi2Hinglish-Swift``) — STT

Both are idempotent: if the model is already cached locally, the call
returns immediately without hitting the network. The script is safe
to re-run on every launch.

Run:
    PYTHONPATH=. python -m scripts.setup_voice_models
"""
from __future__ import annotations

import os
import pathlib
import sys

# Load .env BEFORE reading any IFA_* env vars so the setup phase pulls
# the same models the runtime will load. Without this, e.g. setting
# IFA_WHISPER_MODEL=base.en in .env would still cause the runtime to
# fail with "model not cached" because we'd have pre-downloaded the
# default small.en.
try:
    from dotenv import load_dotenv
    _repo_root = pathlib.Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=_repo_root / ".env", override=False)
except ImportError:
    # First-run path: .env support landed alongside python-dotenv. If
    # the user's venv predates that, fall through silently — they'll
    # just get the defaults this run, which is what they had before.
    pass


def _step(msg: str) -> None:
    print(f"[setup-voice] {msg}", flush=True)


def ensure_openwakeword_models() -> None:
    model_name = os.environ.get("IFA_WAKE_MODEL", "ifa")
    # Skip download if this is a custom .onnx path — those don't live in the cache.
    if os.path.exists(model_name):
        _step(f"wake-word model is a custom path: {model_name}  (no download)")
        return

    _step(f"ensuring openWakeWord model '{model_name}' is cached...")
    from openwakeword.utils import download_models
    download_models(model_names=[model_name])
    _step("openWakeWord ready.")


def ensure_whisper_model() -> None:
    model_name = os.environ.get(
        "IFA_WHISPER_MODEL",
        "Oriserve/Whisper-Hindi2Hinglish-Swift",
    )

    _step(
        f"ensuring transformers Whisper model '{model_name}' is cached..."
    )

    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
    )

    # Download processor files
    AutoProcessor.from_pretrained(model_name)

    # Download model weights
    AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    _step("transformers Whisper ready.")


def main() -> int:
    try:
        ensure_openwakeword_models()
        ensure_whisper_model()
    except Exception as exc:
        _step(f"ERROR: {exc}")
        return 1
    _step("all voice models cached. runtime can now use HF_HUB_OFFLINE=0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
