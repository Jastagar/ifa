"""Input mode switch for the orchestrator.

Default: text mode (``IFA_MODE`` unset or ``text``). Reads ``input("You: ")``.

Voice mode (``IFA_MODE=voice``): wake-word listener + VAD-gated capture +
Whisper transcription, running on a background daemon thread so the main
thread is free to block on ``tts.speak()`` during Ifa's replies without
starving the mic loop.

Topology::

    main thread                   voice thread (daemon)
    -----------                   ---------------------
    get_input()                   sounddevice.InputStream (shared)
      └── queue.get() ─────┐        │
                           │      ┌─┴── wait_for_wake (1280-sample frames)
                           │      └── capture_utterance (512-sample frames)
                           │           └── transcribe_array
                           └────── queue.put(text)
    agent_turn(text, ...)
      └── tts.speak(reply)
          (is_speaking=True, wake-word thread drops its scores)

On any init failure (mic unavailable, models missing, sounddevice import
error), we fall back to text mode with a visible notice — voice mode is
opt-in, crashing the app when it can't start is the wrong move.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Callable, Optional

from ifa.services.tts_service import TTSService


def _text_input() -> str:
    return input("You: ")


def get_input() -> str:
    """Backwards-compatible helper: honors the old in-module ``MODE`` style.

    New code should use ``init_input(tts)`` instead, which sets up the
    voice pipeline once and returns a proper callable.
    """
    mode = os.environ.get("IFA_MODE", "text").lower()
    if mode == "voice":
        # Fallback for callers that haven't migrated: spin up a one-shot.
        voice = VoiceInput(TTSService())
        return voice.get()
    return _text_input()


class VoiceInput:
    """Background wake→capture→transcribe loop feeding a thread-safe queue.

    Construct once at orchestrator startup (after ``TTSService``). Each
    call to ``get()`` blocks until one transcribed utterance arrives.
    ``close()`` stops the mic stream; the daemon thread exits with the
    process anyway.
    """

    def __init__(self, tts_service: TTSService) -> None:
        # Imports are lazy so text-mode users don't pay the voice dep cost.
        import sounddevice as sd

        from ifa.voice.capture import CAPTURE_SAMPLES, capture_utterance
        from ifa.voice.stt import transcribe_array
        from ifa.voice.wake_word import WAKE_CHUNK_SAMPLES, WakeWordListener

        self._listener = WakeWordListener(tts_service=tts_service)
        self._capture_utterance = capture_utterance
        self._transcribe = transcribe_array
        self._wake_chunk_samples = WAKE_CHUNK_SAMPLES
        self._capture_chunk_samples = CAPTURE_SAMPLES

        self._stream = sd.InputStream(
            samplerate=16_000, channels=1, dtype="float32"
        )
        self._stream.start()

        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- public API --

    def start(self) -> None:
        """Spawn the background wake→capture→transcribe daemon thread.

        Kept separate from ``__init__`` so tests can monkey-patch
        ``_listener.wait_for_wake`` / ``_capture_utterance`` /
        ``_transcribe`` before the loop sees them. Production code calls
        ``start()`` right after construction (done by ``init_input``).
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="ifa-voice-loop", daemon=True
        )
        self._thread.start()

    def get(self) -> str:
        """Block until one utterance is transcribed; return its text."""
        return self._queue.get()

    def close(self) -> None:
        """Request the background thread to exit and stop the mic stream."""
        self._stop.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    # -- internals --

    def _read_wake_chunk(self):
        data, _ = self._stream.read(self._wake_chunk_samples)
        return data[:, 0]

    def _read_capture_chunk(self):
        data, _ = self._stream.read(self._capture_chunk_samples)
        return data[:, 0]

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._listener.wait_for_wake(read_chunk=self._read_wake_chunk)
                if self._stop.is_set():
                    return
                audio = self._capture_utterance(read_chunk=self._read_capture_chunk)
                text = self._transcribe(audio)
            except BaseException as exc:
                # BaseException catches test sentinels (which inherit from it)
                # and propagates SystemExit/KeyboardInterrupt cleanly while
                # still shielding Exception subclasses from killing the loop.
                if self._stop.is_set() or not isinstance(exc, Exception):
                    return
                # Surface but don't kill the thread — a bad frame / transient
                # device error shouldn't take voice mode down for the session.
                print(f"[voice] loop error: {exc}")
                continue

            text = (text or "").strip()
            if not text:
                # VAD fired but Whisper heard nothing intelligible — just go
                # back to listening; don't confuse the agent with empty input.
                continue
            self._queue.put(text)


def init_input(tts_service: TTSService) -> Callable[[], str]:
    """Return a ``get_input``-shaped callable based on ``IFA_MODE``.

    Called once at orchestrator startup. In voice mode, spins up
    ``VoiceInput`` (which opens the mic + background thread). On any
    init failure, prints an actionable notice and returns the text-mode
    callable instead — voice is opt-in, so falling back is correct.
    """
    mode = os.environ.get("IFA_MODE", "text").lower()
    if mode != "voice":
        print("[voice] mode=text  (set IFA_MODE=voice to use wake-word input)")
        return _text_input

    try:
        voice = VoiceInput(tts_service=tts_service)
        voice.start()
    except Exception as exc:
        print(
            f"[voice] mode=voice failed to init ({exc}); "
            "falling back to text mode for this session."
        )
        return _text_input

    wake_name = voice._listener.score_key
    whisper_name = os.environ.get("IFA_WHISPER_MODEL", "small.en")
    threshold = voice._listener.threshold
    print(
        f"[voice] mode=voice  wake={wake_name}  whisper={whisper_name}  "
        f"threshold={threshold:.2f}"
    )
    print(f"[voice] say '{wake_name}' to speak; Ctrl-C to quit.")
    return voice.get
