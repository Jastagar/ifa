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
import time
from typing import Optional

from ifa.services.tts_service import TTSService


class _InputMode:
    """Protocol-ish base: every input mode exposes ``get()`` + ``arm_followup()``.

    ``arm_followup`` is a no-op for text mode, but keeps the orchestrator
    loop uniform (it calls it after every tts.speak() regardless of mode).
    """

    def get(self) -> str:
        raise NotImplementedError

    def arm_followup(self) -> None:  # pragma: no cover — trivial
        pass

    def close(self) -> None:  # pragma: no cover — trivial
        pass


class _TextMode(_InputMode):
    def get(self) -> str:
        return input("You: ")


def get_input() -> str:
    """Backwards-compatible helper kept for any lingering callers.

    New code should use ``init_input(tts)`` which returns an ``_InputMode``
    object supporting both ``get()`` and ``arm_followup()``.
    """
    mode = os.environ.get("IFA_MODE", "text").lower()
    if mode == "voice":
        voice = VoiceInput(TTSService())
        voice.start()
        return voice.get()
    return _TextMode().get()


class VoiceInput(_InputMode):
    """Background wake→capture→transcribe loop feeding a thread-safe queue.

    Construct once at orchestrator startup (after ``TTSService``). Each
    call to ``get()`` blocks until one transcribed utterance arrives.
    ``close()`` stops the mic stream; the daemon thread exits with the
    process anyway.

    Follow-up window: after Ifa speaks a reply, the orchestrator calls
    ``arm_followup()``. For ``IFA_FOLLOWUP_WINDOW_SEC`` seconds
    afterwards the voice loop skips the wake-word step and goes
    straight to capture — so a conversational follow-up (“what about
    tomorrow?”) just works. If the user doesn't speak within that
    window, the deadline expires and the next turn requires the wake
    word again.
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

        # Follow-up window state
        self._followup_window_sec = float(
            os.environ.get("IFA_FOLLOWUP_WINDOW_SEC", "5")
        )
        self._followup_until = 0.0  # monotonic deadline; 0 = no follow-up armed

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

    def arm_followup(self) -> None:
        """Mark the next capture as follow-up — skip wake-word if within window.

        Called by the orchestrator right after ``tts.speak(reply)`` returns.
        """
        if self._followup_window_sec > 0:
            self._followup_until = time.monotonic() + self._followup_window_sec

    def _in_followup_window(self) -> bool:
        return time.monotonic() < self._followup_until

    # -- internals --

    def _read_wake_chunk(self):
        data, _ = self._stream.read(self._wake_chunk_samples)
        return data[:, 0]

    def _read_capture_chunk(self):
        data, _ = self._stream.read(self._capture_chunk_samples)
        return data[:, 0]

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            in_followup = self._in_followup_window()
            # Clear the arm now so a failed/empty follow-up doesn't bleed
            # into subsequent iterations.
            self._followup_until = 0.0
            try:
                if not in_followup:
                    print("[voice] waiting for wake word...")
                    self._listener.wait_for_wake(
                        read_chunk=self._read_wake_chunk
                    )
                    if self._stop.is_set():
                        return
                    print("[voice] wake detected — capturing...")
                    capture_kwargs = {}
                else:
                    print(
                        f"[voice] follow-up ({self._followup_window_sec:g}s) "
                        "— listening without wake word..."
                    )
                    capture_kwargs = {
                        "start_timeout_ms": int(
                            self._followup_window_sec * 1000
                        )
                    }

                audio = self._capture_utterance(
                    read_chunk=self._read_capture_chunk, **capture_kwargs
                )
                print(f"[voice] captured {len(audio) / 16_000:.2f}s — transcribing...")
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
                print("[voice] (no speech detected / empty transcript)")
                continue
            print(f"[voice] heard: {text!r}")
            self._queue.put(text)


def init_input(tts_service: TTSService) -> _InputMode:
    """Return an ``_InputMode`` object based on ``IFA_MODE``.

    Called once at orchestrator startup. In voice mode, spins up
    ``VoiceInput`` (which opens the mic + background thread). On any
    init failure, prints an actionable notice and returns a text-mode
    stub instead — voice is opt-in, so falling back is correct.

    The returned object exposes ``get()`` (one turn of user input) and
    ``arm_followup()`` (no-op in text mode; primes the follow-up
    window in voice mode).
    """
    mode = os.environ.get("IFA_MODE", "text").lower()
    if mode != "voice":
        print("[voice] mode=text  (set IFA_MODE=voice to use wake-word input)")
        return _TextMode()

    try:
        voice = VoiceInput(tts_service=tts_service)
        voice.start()
    except Exception as exc:
        print(
            f"[voice] mode=voice failed to init ({exc}); "
            "falling back to text mode for this session."
        )
        return _TextMode()

    wake_name = voice._listener.score_key
    whisper_name = os.environ.get("IFA_WHISPER_MODEL", "small.en")
    threshold = voice._listener.threshold
    followup_sec = voice._followup_window_sec
    print(
        f"[voice] mode=voice  wake={wake_name}  whisper={whisper_name}  "
        f"threshold={threshold:.2f}  followup={followup_sec:g}s"
    )
    print(f"[voice] say '{wake_name}' to speak; Ctrl-C to quit.")
    return voice
