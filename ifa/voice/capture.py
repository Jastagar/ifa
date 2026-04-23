"""Mic capture + VAD end-of-turn for Stage 2 voice mode.

After the wake-word fires (Unit 2), we record until the user stops
speaking. This module provides ``capture_utterance`` which reads
512-sample chunks from an injected callable, feeds them to Silero VAD
(the ONNX variant bundled inside faster-whisper, no torch dep), and
returns a concatenated float32 numpy array once the user has been
silent for ``IFA_VAD_SILENCE_MS`` milliseconds — or once
``IFA_VAD_MAX_UTTERANCE_MS`` has elapsed, whichever comes first.

Tunables (all env vars):
  IFA_VAD_SILENCE_MS        (default 1500)  -- trailing-silence threshold
  IFA_VAD_MAX_UTTERANCE_MS  (default 30000) -- hard cap per turn
  IFA_VAD_THRESHOLD         (default 0.5)   -- per-chunk speech probability
                                              at/above which the chunk is
                                              "speech" for the silence timer

The VAD call is wrapped in try/except: if
``faster_whisper.vad.SileroVADModel`` breaks on a future faster-whisper
upgrade (internal API, not guaranteed stable), we fall back to a
simple RMS energy threshold — degraded but still functional.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

import numpy as np

# silero_vad_v6.onnx is trained at 16 kHz with 512-sample input blocks.
CAPTURE_SAMPLES = 512
SAMPLE_RATE = 16_000
CHUNK_MS = CAPTURE_SAMPLES * 1000 // SAMPLE_RATE  # 32 ms


def _resolve_vad_path() -> str:
    """Find the silero_vad_v6.onnx bundled inside faster-whisper's assets dir."""
    import faster_whisper.assets
    import pathlib
    asset = (
        pathlib.Path(faster_whisper.assets.__path__[0]) / "silero_vad_v6.onnx"
    )
    if not asset.exists():
        raise FileNotFoundError(
            f"Expected Silero VAD at {asset}. faster-whisper install looks broken."
        )
    return str(asset)


def _load_vad():
    """Construct the SileroVADModel lazily (onnxruntime import + model load cost)."""
    from faster_whisper.vad import SileroVADModel
    return SileroVADModel(_resolve_vad_path())


def _speech_prob_via_vad(vad, chunk_f32: np.ndarray) -> float:
    """Run Silero VAD on a 512-sample float32 chunk; return prob in [0, 1]."""
    # Returns a numpy array of shape (1,) — one probability per 512-sample block.
    arr = vad(chunk_f32)
    return float(arr[0])


def _speech_prob_via_energy(chunk_f32: np.ndarray) -> float:
    """Fallback: crude RMS-based voice-activity decision.

    Returns a pseudo-probability in [0, 1] so the caller's threshold
    comparison still works. RMS above ~0.02 in float32 typically means
    speech; below means silence.
    """
    rms = float(np.sqrt(np.mean(chunk_f32.astype(np.float32) ** 2)))
    # Map rms to ~[0, 1] with 0.05 as the inflection point.
    return min(1.0, rms / 0.05)


def capture_utterance(
    read_chunk: Callable[[], np.ndarray],
    *,
    silence_ms: Optional[int] = None,
    max_utterance_ms: Optional[int] = None,
    threshold: Optional[float] = None,
    vad: Optional[object] = None,
) -> np.ndarray:
    """Capture one utterance: return audio up to the end of speech.

    ``read_chunk`` must yield 1D float32 arrays of exactly
    ``CAPTURE_SAMPLES`` samples at 16 kHz.

    Returns a concatenated float32 array covering the entire utterance
    including the trailing silence window (we don't trim — Whisper
    handles trailing silence fine, and trimming would mask
    late-utterance trailing sounds).
    """
    if silence_ms is None:
        silence_ms = int(os.environ.get("IFA_VAD_SILENCE_MS", "1500"))
    if max_utterance_ms is None:
        max_utterance_ms = int(os.environ.get("IFA_VAD_MAX_UTTERANCE_MS", "30000"))
    if threshold is None:
        threshold = float(os.environ.get("IFA_VAD_THRESHOLD", "0.5"))

    # Lazy VAD load: tests inject ``vad`` directly, real callers get the
    # bundled model on first capture.
    vad_model = vad if vad is not None else _load_vad()
    vad_failed = False  # flips on first exception from the VAD

    buffer: list[np.ndarray] = []
    silence_elapsed_ms = 0
    start = time.monotonic()
    saw_speech = False

    while True:
        chunk = read_chunk()
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        buffer.append(chunk)

        if not vad_failed:
            try:
                prob = _speech_prob_via_vad(vad_model, chunk)
            except Exception:
                vad_failed = True
                prob = _speech_prob_via_energy(chunk)
        else:
            prob = _speech_prob_via_energy(chunk)

        if prob >= threshold:
            silence_elapsed_ms = 0
            saw_speech = True
        else:
            # Only start counting silence after the user has started speaking —
            # otherwise an early 1.5s of hesitation cuts the turn before a word.
            if saw_speech:
                silence_elapsed_ms += CHUNK_MS
                if silence_elapsed_ms >= silence_ms:
                    break

        if (time.monotonic() - start) * 1000 >= max_utterance_ms:
            break

    return np.concatenate(buffer) if buffer else np.zeros(0, dtype=np.float32)
