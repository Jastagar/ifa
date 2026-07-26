"""Small, non-speech audio cues used by voice mode."""
from __future__ import annotations

import os
from pathlib import Path
import wave

import numpy as np


_LISTENING_CUE_PATH = Path(__file__).resolve().parent.parent / "audios" / "wakesound.wav"


def play_listening_cue() -> None:
    """Play the bundled wake sound to acknowledge that Ifa is listening.

    Set ``IFA_LISTENING_CUE=0`` to silence the cue. Audio failures are
    deliberately non-fatal: voice capture must keep working even when the
    output device is unavailable or busy.
    """
    if os.environ.get("IFA_LISTENING_CUE", "1").lower() in {"0", "false", "no", "off"}:
        return

    try:
        import sounddevice as sd

        with wave.open(str(_LISTENING_CUE_PATH), "rb") as cue:
            if cue.getcomptype() != "NONE" or cue.getsampwidth() != 2:
                raise ValueError("wakesound.wav must be an uncompressed 16-bit PCM WAV")
            sample_rate = cue.getframerate()
            channels = cue.getnchannels()
            audio = np.frombuffer(cue.readframes(cue.getnframes()), dtype=np.int16)

        audio = (audio.reshape(-1, channels).astype(np.float32) / 32768.0)
        sd.play(audio, sample_rate, blocking=True)
    except Exception as exc:
        print(f"[voice] listening cue unavailable: {exc}")
