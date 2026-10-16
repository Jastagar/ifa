from __future__ import annotations

import os
import pathlib
from typing import Callable, Optional, Protocol

import numpy as np

_DEFAULT_MODEL = str(
    pathlib.Path(__file__).resolve().parent.parent / "models" / "ifa.onnx"
)

_FALLBACK_MODEL = "ifa"

WAKE_CHUNK_SAMPLES = 1280


def _resolve_model_spec() -> str:
    return os.environ.get("IFA_WAKE_MODEL", _DEFAULT_MODEL)


def _is_path_spec(spec: str) -> bool:
    return (
        "/" in spec
        or "\\" in spec
        or spec.endswith(".onnx")
        or spec.endswith(".tflite")
    )


def _derive_score_key(model_spec: str) -> str:
    if os.path.exists(model_spec):
        return os.path.splitext(os.path.basename(model_spec))[0]
    return model_spec


class WakeWordInitError(RuntimeError):
    pass


class _TTSProtocol(Protocol):
    @property
    def is_speaking(self) -> bool: ...


class WakeWordListener:
    def __init__(
        self,
        tts_service: Optional[_TTSProtocol] = None,
        *,
        threshold: Optional[float] = None,
        model_spec: Optional[str] = None,
    ) -> None:
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise WakeWordInitError(
                "openwakeword is not installed. Run `pip install openwakeword==0.6.0` "
                "or use text mode (IFA_MODE=text)."
            ) from exc

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
        self._consecutive_required = int(
            os.environ.get("IFA_WAKE_CONSECUTIVE", "2")
        )
        self._tts = tts_service

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def score_key(self) -> str:
        return self._score_key

    @property
    def fallback_from(self) -> Optional[str]:
        return self._fallback_from

    def wait_for_wake(
        self,
        read_chunk: Callable[[], np.ndarray],
        interrupt: Optional[Callable[[], bool]] = None,
    ) -> Optional[float]:
        consecutive = 0
        last_score = 0.0
        silence = np.zeros(WAKE_CHUNK_SAMPLES, dtype=np.int16)
        while True:
            chunk = read_chunk()
            if self._tts is not None and self._tts.is_speaking:
                self._model.predict(silence)
                consecutive = 0
                continue
            if interrupt is not None and interrupt():
                self._model.reset()
                return None
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
    if chunk.dtype == np.int16:
        return chunk
    scaled = np.clip(chunk * 32768.0, -32768.0, 32767.0)
    return scaled.astype(np.int16)
