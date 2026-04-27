"""Wake-word listener for Stage 2 voice mode.

Continuously scores 16 kHz mono audio frames against an openWakeWord
model (default ``hey_mycroft``, configurable via ``IFA_WAKE_MODEL`` —
either a built-in name or a path to a custom ``.onnx``); returns as
soon as the confidence score crosses a threshold. Designed to run on
a background daemon thread spawned by ``VoiceInput`` (Unit 5); the
main thread is busy running ``agent_turn`` + TTS.

Mute cooperation: on each frame, the listener reads ``tts.is_speaking``
(see ``ifa.services.tts_service.TTSService``). While Ifa is speaking (or
in the post-TTS cooldown window), frames are dropped without scoring
so the mic can't self-trigger on Ifa's own voice.

Stream injection: ``wait_for_wake`` takes a ``read_chunk`` callable
rather than owning a ``sounddevice.InputStream`` directly. The caller
(VoiceInput) opens one long-lived stream shared with Unit 3's capture
step and passes a thin closure. This keeps the unit testable without
sounddevice and keeps the same stream live across wake → capture → wake
cycles.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Protocol

import numpy as np

# Default wake-word model. Swap via IFA_WAKE_MODEL env var.
#
# Progression of defaults during Stage 2 development:
#   1. ``hey_jarvis`` — original plan; hit 3/50 on this repo's owner's
#      voice.
#   2. ``alexa`` — swapped in thinking the larger training set would
#      help; still failed on the same voice (0/20, max score 0.02).
#   3. ``hey_mycroft`` — current default. Offline scored 1.0 on an
#      8-second recording of the owner saying the wake word, vs. 0.02
#      for alexa on the same voice. Mycroft's training data apparently
#      fits better.
#
# Identity note: the "hey_" prefix is a compromise — the user would
# prefer a prefix-free name. Custom-trained "hey ifa" (or just "ifa")
# is the follow-up; it plugs in here with zero code change by pointing
# IFA_WAKE_MODEL at a custom .onnx file.
#
# IFA_WAKE_MODEL accepts either a built-in name (``alexa``,
# ``hey_jarvis``, ``hey_mycroft``, ``hey_rhasspy``) or a filesystem
# path to a custom .onnx model.
_DEFAULT_MODEL = "hey_mycroft"

# Where the listener falls back to if a path-style spec points at a
# missing file (Stage 3 prep). Keep this in sync with what
# scripts/setup_voice_models.py knows how to download. The fallback is
# noisy on purpose — see the WARNING in __init__.
_FALLBACK_MODEL = "hey_mycroft"

# openWakeWord's expected frame size: 80 ms at 16 kHz.
WAKE_CHUNK_SAMPLES = 1280


def _resolve_model_spec() -> str:
    """Read ``IFA_WAKE_MODEL`` from the environment, falling back to ``_DEFAULT_MODEL``.

    Pure env lookup — no existence check. The path-vs-name detection
    and missing-file fallback live in ``WakeWordListener.__init__`` so
    they run per-construction (per the Stage 3 plan: existence-check
    happens at listener-construction time, not module-import time).
    """
    return os.environ.get("IFA_WAKE_MODEL", _DEFAULT_MODEL)


def _is_path_spec(spec: str) -> bool:
    """Return True if ``spec`` looks like a filesystem path, False if it's a built-in name.

    Built-in openWakeWord model names are short tokens like ``alexa``,
    ``hey_jarvis``, ``hey_mycroft``. None contain a path separator or a
    file extension. Paths to custom ``.onnx`` models do.
    """
    return (
        "/" in spec
        or "\\" in spec
        or spec.endswith(".onnx")
        or spec.endswith(".tflite")
    )


def _derive_score_key(model_spec: str) -> str:
    """Score-dict key that ``Model.predict`` uses for a given spec.

    - Built-in name: the name is the key (``alexa`` → ``alexa``)
    - File path: basename without extension (``/x/ifa.onnx`` → ``ifa``)
    """
    if os.path.exists(model_spec):
        return os.path.splitext(os.path.basename(model_spec))[0]
    return model_spec


class WakeWordInitError(RuntimeError):
    """Raised when the wake-word model can't be loaded (download failure, missing file, etc.)."""


class _TTSProtocol(Protocol):
    @property
    def is_speaking(self) -> bool: ...


class WakeWordListener:
    """Listens for the configured wake word on a stream of audio chunks.

    Parameters
    ----------
    tts_service:
        Any object exposing an ``is_speaking`` bool property. The wake
        loop drops frames while it's True.
    threshold:
        Override for the detection threshold. Defaults to
        ``IFA_WAKE_THRESHOLD`` env var (falling back to 0.7).
    model_spec:
        Override for the model spec — either a built-in name or a path
        to a custom ``.onnx``. Defaults to the value of
        ``IFA_WAKE_MODEL`` (built-in ``hey_mycroft`` if unset).
    """

    def __init__(
        self,
        tts_service: Optional[_TTSProtocol] = None,
        *,
        threshold: Optional[float] = None,
        model_spec: Optional[str] = None,
    ) -> None:
        # Imported lazily so the rest of Ifa can import this module even
        # when openwakeword isn't installed (text mode shouldn't pay for
        # the voice deps).
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:  # pragma: no cover
            raise WakeWordInitError(
                "openwakeword is not installed. Run `pip install openwakeword==0.6.0` "
                "or use text mode (IFA_MODE=text)."
            ) from exc

        # Resolve spec, then handle missing-file fallback per-construction
        # (NOT at module-import time — see _resolve_model_spec docstring).
        # If the resolved spec looks like a filesystem path AND that file
        # is absent, log a prominent WARNING and substitute the built-in
        # fallback model. This catches the post-Stage-3 case where the
        # bundled ifa.onnx is somehow not on disk (failed install, fresh
        # clone with corrupted blob, etc.) — without it, the listener
        # would try to download_models() with the path string as a name
        # and crash with a confusing error.
        raw_spec = model_spec or _resolve_model_spec()
        self._fallback_from: Optional[str] = None
        if _is_path_spec(raw_spec) and not os.path.exists(raw_spec):
            print(
                f"[wake_word] WARNING: wake-word model not found at "
                f"{raw_spec!r}; falling back to built-in {_FALLBACK_MODEL!r}. "
                f"Wake word will respond to '{_FALLBACK_MODEL}' until the "
                f"missing file is restored. Re-run scripts/setup_voice_models "
                f"or restore the file at the expected path.",
                flush=True,
            )
            self._fallback_from = raw_spec
            self._model_spec = _FALLBACK_MODEL
        else:
            self._model_spec = raw_spec
        self._score_key = _derive_score_key(self._model_spec)

        try:
            # Only download built-in names; custom .onnx paths are already on disk.
            if not os.path.exists(self._model_spec):
                download_models(model_names=[self._model_spec])
            self._model = Model(
                wakeword_models=[self._model_spec],
                inference_framework="onnx",
            )
        except Exception as exc:
            raise WakeWordInitError(
                f"Failed to initialize '{self._model_spec}' wake-word model: {exc}. "
                "Ensure the machine has internet for first-run download of built-in "
                "models; subsequent runs use the cached model. For custom .onnx paths, "
                "verify the file exists."
            ) from exc

        if threshold is None:
            threshold = float(os.environ.get("IFA_WAKE_THRESHOLD", "0.7"))
        self._threshold = threshold
        # Require N consecutive frames above threshold before firing. Kills
        # single-frame score spikes on ambient noise. Each frame is 80 ms,
        # so 2 frames = 160 ms of sustained signal above threshold.
        self._consecutive_required = int(
            os.environ.get("IFA_WAKE_CONSECUTIVE", "2")
        )
        self._tts = tts_service

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def score_key(self) -> str:
        """The dict key under which scores arrive from ``Model.predict``."""
        return self._score_key

    @property
    def fallback_from(self) -> Optional[str]:
        """Original spec that triggered missing-file fallback, or ``None`` if no fallback fired.

        Surfaced so the launcher startup line can distinguish
        ``wake=ifa (bundled)`` from ``wake=hey_mycroft (FALLBACK from
        ifa.onnx)`` — without this, a missing-file fallback is silent
        and a user sees a wake-word that doesn't match the configured
        model with no obvious explanation.
        """
        return self._fallback_from

    def wait_for_wake(
        self,
        read_chunk: Callable[[], np.ndarray],
    ) -> float:
        """Block until a wake-word detection; return the detection score.

        ``read_chunk`` must return a 1D numpy array of ``WAKE_CHUNK_SAMPLES``
        samples of 16 kHz mono audio. float32 (``[-1.0, 1.0]``) or int16
        are both accepted; float32 is converted to int16 PCM before
        scoring.

        Fires only after ``self._consecutive_required`` consecutive frames
        score above threshold — this suppresses single-frame spikes on
        ambient noise while barely affecting wake-word latency (each extra
        required frame is 80 ms).

        Mute handling: while ``tts.is_speaking`` we feed int16 **silence**
        to the model instead of the live mic chunk. This keeps the
        AudioFeatures rolling buffer advancing in real time (skipping
        predict entirely leaves stale wake-word context in the buffer,
        which re-fires as soon as a fresh chunk slides in) AND prevents
        Ifa's own TTS from poisoning the buffer via mic bleed-over. We
        still drop the returned scores during mute.

        After a successful detection we call ``self._model.reset()`` to
        clear the prediction buffer, so the next wait_for_wake starts
        from a clean slate.
        """
        consecutive = 0
        last_score = 0.0
        silence = np.zeros(WAKE_CHUNK_SAMPLES, dtype=np.int16)
        while True:
            chunk = read_chunk()
            if self._tts is not None and self._tts.is_speaking:
                # Keep the feature buffer flowing with silence; ignore score.
                self._model.predict(silence)
                consecutive = 0
                continue
            chunk_i16 = _to_int16(chunk)
            scores = self._model.predict(chunk_i16)
            score = float(scores.get(self._score_key, 0.0))
            if score >= self._threshold:
                consecutive += 1
                last_score = score
                if consecutive >= self._consecutive_required:
                    self._model.reset()
                    return last_score
            else:
                consecutive = 0


def _to_int16(chunk: np.ndarray) -> np.ndarray:
    """Normalize a mic chunk to int16 PCM for openWakeWord."""
    if chunk.dtype == np.int16:
        return chunk
    # sounddevice float32 gives values in [-1.0, 1.0]
    scaled = np.clip(chunk * 32768.0, -32768.0, 32767.0)
    return scaled.astype(np.int16)
