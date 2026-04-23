"""Speech-to-text via faster-whisper.

Primary entry point for Stage 2 voice mode is ``transcribe_array`` —
Unit 3's capture step produces a float32 numpy array at 16 kHz, and
this module runs it through Whisper and returns the transcribed text.

Model is loaded lazily the first time a transcribe function is called,
then reused across turns. Configurable via ``IFA_WHISPER_MODEL`` env
var; defaults to ``small.en`` (good-quality English, ~470MB, fits
comfortably on an M4 Pro or an RTX 4060Ti).

``compute_type='int8'`` keeps memory + latency manageable on CPU while
matching ``small.en``'s accuracy tolerance.

``transcribe(audio_path)`` is kept for backwards compatibility with
``ifa.voice.input`` until Unit 5 replaces that file-mode shim.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

_model: Optional[object] = None  # faster_whisper.WhisperModel, loaded lazily


def _get_model() -> object:
    """Load (or return) the singleton WhisperModel.

    Device selection (``IFA_WHISPER_DEVICE``):
      - ``auto`` (default): try CUDA first (float16); fall back to CPU (int8)
      - ``cuda``: CUDA (float16); raises if unavailable
      - ``cpu``: CPU (int8)

    CUDA requires ``nvidia-cublas-cu12`` + ``nvidia-cudnn-cu12`` (shipped
    on Windows/Linux via requirements.txt); on macOS we stay on CPU.
    """
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        model_name = os.environ.get("IFA_WHISPER_MODEL", "small.en")
        device = os.environ.get("IFA_WHISPER_DEVICE", "auto").lower()

        # NOTE: first arg is ``model_size_or_path`` (positional). There
        # is no keyword ``size=``; passing size= as kwarg raises TypeError.
        if device in ("auto", "cuda"):
            try:
                _model = WhisperModel(
                    model_name, device=device, compute_type="float16"
                )
                print(f"[whisper] loaded {model_name} on {device} (float16)")
                return _model
            except Exception as exc:
                if device == "cuda":
                    # Explicit cuda request with no CUDA → fail loudly
                    raise
                print(
                    f"[whisper] GPU init failed ({exc}); falling back to CPU int8. "
                    "On Windows/Linux with an Nvidia GPU, install "
                    "nvidia-cublas-cu12 + nvidia-cudnn-cu12 to enable CUDA."
                )
        _model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"[whisper] loaded {model_name} on cpu (int8)")
    return _model


def transcribe_array(audio: np.ndarray, *, language: str = "en") -> str:
    """Transcribe a float32 numpy array of 16 kHz mono audio.

    Returns the concatenated segment text with leading/trailing whitespace
    stripped. If Whisper raises (rare, but possible on corrupted input
    or model-load failures mid-session), returns an empty string so the
    agent loop treats the turn as a no-op rather than crashing.
    """
    if audio is None or len(audio) == 0:
        return ""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    try:
        model = _get_model()
        segments, _info = model.transcribe(
            audio,
            language=language,
            vad_filter=False,  # we already VAD'd in Unit 3
            beam_size=5,
        )
        # segments is a generator; materialize and join
        text = " ".join(seg.text for seg in segments)
        return text.strip()
    except RuntimeError:
        return ""


def transcribe(audio_path: str) -> str:
    """File-path variant — reuses the same lazy-loaded model.

    Kept for the transitional ``ifa.voice.input.get_audio_input`` shim.
    Once Unit 5 replaces that with ``VoiceInput.get`` calling
    ``transcribe_array`` directly, this function can be removed.
    """
    try:
        model = _get_model()
        segments, _info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text for seg in segments)
        return text.strip()
    except RuntimeError:
        return ""
