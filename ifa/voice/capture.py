from __future__ import annotations

import os
import time
from typing import Callable, Optional

import numpy as np

CAPTURE_SAMPLES = 512
SAMPLE_RATE = 16_000
CHUNK_MS = CAPTURE_SAMPLES * 1000 // SAMPLE_RATE


def _resolve_vad_path() -> str:
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
    from faster_whisper.vad import SileroVADModel
    return SileroVADModel(_resolve_vad_path())


def _speech_prob_via_vad(vad, chunk_f32: np.ndarray) -> float:
    arr = vad(chunk_f32)
    return float(arr[0])


def _speech_prob_via_energy(chunk_f32: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(chunk_f32.astype(np.float32) ** 2)))
    return min(1.0, rms / 0.05)


def capture_utterance(
    read_chunk: Callable[[], np.ndarray],
    *,
    silence_ms: Optional[int] = None,
    max_utterance_ms: Optional[int] = None,
    threshold: Optional[float] = None,
    start_timeout_ms: Optional[int] = None,
    vad: Optional[object] = None,
) -> np.ndarray:
    if silence_ms is None:
        silence_ms = int(os.environ.get("IFA_VAD_SILENCE_MS", "1500"))
    if max_utterance_ms is None:
        max_utterance_ms = int(os.environ.get("IFA_VAD_MAX_UTTERANCE_MS", "30000"))
    if threshold is None:
        threshold = float(os.environ.get("IFA_VAD_THRESHOLD", "0.5"))

    vad_model = vad if vad is not None else _load_vad()
    vad_failed = False

    buffer: list[np.ndarray] = []
    silence_elapsed_ms = 0
    start = time.monotonic()
    saw_speech = False

    while True:
        chunk = read_chunk()
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        if not vad_failed:
            try:
                prob = _speech_prob_via_vad(vad_model, chunk)
            except Exception:
                vad_failed = True
                prob = _speech_prob_via_energy(chunk)
        else:
            prob = _speech_prob_via_energy(chunk)

        if prob >= threshold:
            saw_speech = True

        elapsed_ms = (time.monotonic() - start) * 1000

        if (
            start_timeout_ms is not None
            and not saw_speech
            and elapsed_ms >= start_timeout_ms
        ):
            return np.zeros(0, dtype=np.float32)

        buffer.append(chunk)

        if prob >= threshold:
            silence_elapsed_ms = 0
        else:
            if saw_speech:
                silence_elapsed_ms += CHUNK_MS
                if silence_elapsed_ms >= silence_ms:
                    break

        if elapsed_ms >= max_utterance_ms:
            break

    return np.concatenate(buffer) if buffer else np.zeros(0, dtype=np.float32)
