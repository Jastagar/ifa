from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Optional

from ifa.services.tts_service import TTSService
from ifa.utils import latency


class _InputMode:
    def get(self) -> str:
        raise NotImplementedError

    def arm_followup(self) -> None:
        pass

    def pause_while(self, check: Callable[[], bool]) -> None:
        pass

    def end_turn(self) -> None:
        pass

    def close(self) -> None:
        pass


class _TextMode(_InputMode):
    def get(self) -> str:
        return input("You: ")


class VoiceInput(_InputMode):
    def __init__(
        self,
        tts_service: TTSService,
        notifications=None,
        announce_urgent: Optional[Callable[[], None]] = None,
        speaker_check=None,
    ) -> None:
        import sounddevice as sd

        from ifa.voice import speaker

        from ifa.voice.capture import CAPTURE_SAMPLES, capture_utterance
        from ifa.voice.cues import play_listening_cue
        from ifa.voice.stt import transcribe_array
        from ifa.voice.wake_word import WAKE_CHUNK_SAMPLES, WakeWordListener

        self._tts = tts_service
        self._notifications = notifications
        self._announce_urgent = announce_urgent
        self._notification_listen_sec = float(
            os.environ.get("IFA_NOTIF_LISTEN_SEC", "8")
        )
        self._listener = WakeWordListener(tts_service=tts_service)
        self._capture_utterance = capture_utterance
        self._play_listening_cue = play_listening_cue
        self._transcribe = transcribe_array
        self._speaker = speaker_check or speaker.shared()
        self._wake_chunk_samples = WAKE_CHUNK_SAMPLES
        self._capture_chunk_samples = CAPTURE_SAMPLES

        self._stream = sd.InputStream(
            samplerate=16_000, channels=1, dtype="float32"
        )
        self._stream.start()

        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._followup_window_sec = float(
            os.environ.get("IFA_FOLLOWUP_WINDOW_SEC", "5")
        )
        self._followup_until = 0.0
        self._turn_complete = threading.Event()
        self._paused: Callable[[], bool] = lambda: False


    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="ifa-voice-loop", daemon=True
        )
        self._thread.start()

    def get(self) -> str:
        return self._queue.get()

    def close(self) -> None:
        self._stop.set()
        self._turn_complete.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def pause_while(self, check: Callable[[], bool]) -> None:
        self._paused = check

    def end_turn(self) -> None:
        self._followup_until = 0.0
        self._turn_complete.set()

    def arm_followup(self) -> None:
        if self._followup_window_sec > 0:
            self._followup_until = time.monotonic() + self._followup_window_sec
        self._turn_complete.set()

    def _in_followup_window(self) -> bool:
        return time.monotonic() < self._followup_until


    def _notification_waiting(self) -> bool:
        return (
            self._notifications is not None
            and self._notifications.announcement_ready.is_set()
        )

    def _wait_for_tts_silence(self) -> bool:
        while self._tts is not None and self._tts.is_speaking:
            if self._stop.is_set():
                return False
            time.sleep(0.05)
        return True

    def _drain_stream(self) -> None:
        try:
            available = int(self._stream.read_available)
        except Exception:
            return
        if available <= 0:
            return
        try:
            self._stream.read(available)
        except Exception:
            pass

    def _maybe_dump_wav(self, audio) -> None:
        out_dir = os.environ.get("IFA_VOICE_DEBUG_WAV")
        if not out_dir:
            return
        try:
            import wave
            import numpy as _np
            os.makedirs(out_dir, exist_ok=True)
            ts = int(time.monotonic() * 1000)
            path = os.path.join(out_dir, f"ifa_capture_{ts}.wav")
            audio_i16 = _np.clip(audio * 32768.0, -32768.0, 32767.0).astype(_np.int16)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16_000)
                wf.writeframes(audio_i16.tobytes())
            print(f"[voice] dumped WAV: {path}")
        except Exception as exc:
            print(f"[voice] WAV dump failed: {exc}")

    def _read_wake_chunk(self):
        data, _ = self._stream.read(self._wake_chunk_samples)
        return data[:, 0]

    def _read_capture_chunk(self):
        data, _ = self._stream.read(self._capture_chunk_samples)
        return data[:, 0]

    def _run_loop(self) -> None:
        needs_turn_wait = False
        was_paused = False
        while not self._stop.is_set():
            if needs_turn_wait:
                while not self._stop.is_set():
                    if self._turn_complete.wait(timeout=0.5):
                        break
                if self._stop.is_set():
                    return
                self._turn_complete.clear()
                needs_turn_wait = False

            if self._paused():
                if not was_paused:
                    print("[voice] phone connected: PC mic paused")
                    was_paused = True
                try:
                    self._read_wake_chunk()
                except Exception:
                    time.sleep(0.08)
                continue
            if was_paused:
                print("[voice] phone disconnected: PC mic listening again")
                was_paused = False

            in_followup = self._in_followup_window()
            self._followup_until = 0.0
            announcement = (
                self._notifications.take_announcement() if self._notifications else None
            )
            try:
                play_cue = True
                if announcement:
                    if not self._wait_for_tts_silence():
                        return
                    print(f"[voice] notification ({announcement}) — listening without wake word...")
                    if announcement == "urgent" and self._announce_urgent:
                        self._announce_urgent()
                    capture_kwargs = {
                        "start_timeout_ms": int(self._notification_listen_sec * 1000)
                    }
                elif not in_followup:
                    print("[voice] waiting for wake word...")
                    score = self._listener.wait_for_wake(
                        read_chunk=self._read_wake_chunk,
                        interrupt=lambda: self._notification_waiting() or self._paused(),
                    )
                    if self._stop.is_set():
                        return
                    if score is None:
                        continue
                    print("[voice] wake detected — capturing...")
                    capture_kwargs = {}
                else:
                    play_cue = False
                    print(
                        f"[voice] follow-up ({self._followup_window_sec:g}s) "
                        "— listening without wake word..."
                    )
                    capture_kwargs = {
                        "start_timeout_ms": int(
                            self._followup_window_sec * 1000
                        )
                    }

                if not self._wait_for_tts_silence():
                    return
                if play_cue:
                    self._play_listening_cue()
                self._drain_stream()

                audio = self._capture_utterance(
                    read_chunk=self._read_capture_chunk, **capture_kwargs
                )
                silence_ms = int(os.environ.get("IFA_VAD_SILENCE_MS", "1500"))
                timer = latency.begin(
                    "voice",
                    t0=time.perf_counter() - silence_ms / 1000,
                    vad_wait_ms=silence_ms,
                )
                timer.mark("captured")
                duration = len(audio) / 16_000

                if len(audio):
                    import numpy as _np
                    peak = float(_np.abs(audio).max())
                    rms = float(_np.sqrt(_np.mean(audio.astype(_np.float32) ** 2)))
                    print(
                        f"[voice] captured {duration:.2f}s "
                        f"(peak={peak:.3f} rms={rms:.3f}) — transcribing..."
                    )
                    self._maybe_dump_wav(audio)
                else:
                    print(f"[voice] captured {duration:.2f}s — transcribing...")

                if len(audio):
                    verdict = self._speaker.check(audio, "pc")
                    if not verdict.accept:
                        print(f"[voice] ignored: {verdict.describe(self._speaker.threshold)}")
                        continue
                    if verdict.score is not None:
                        print(f"[voice] {verdict.describe(self._speaker.threshold)}")

                transcribedText = self._transcribe(audio)
                timer.mark("stt_done")

            except BaseException as exc:
                if self._stop.is_set() or not isinstance(exc, Exception):
                    return
                print(f"[voice] loop error: {exc}")
                continue

            transcribedText = (transcribedText or "").strip()
            if not transcribedText:
                print("[voice] (no speech detected / empty transcript)")
                continue
            print(f"[voice] heard: {transcribedText!r}")
            self._queue.put(transcribedText)
            needs_turn_wait = True


def init_input(
    tts_service: TTSService,
    notifications=None,
    announce_urgent: Optional[Callable[[], None]] = None,
) -> _InputMode:
    mode = os.environ.get("IFA_MODE", "text").lower()
    if mode != "voice":
        print("[voice] mode=text  (set IFA_MODE=voice to use wake-word input)")
        return _TextMode()

    try:
        voice = VoiceInput(
            tts_service=tts_service,
            notifications=notifications,
            announce_urgent=announce_urgent,
        )
        voice.start()
    except Exception as exc:
        print(
            f"[voice] mode=voice failed to init ({exc}); "
            "falling back to text mode for this session."
        )
        return _TextMode()

    wake_name = voice._listener.score_key
    fallback_from = voice._listener.fallback_from
    whisper_name = os.environ.get("IFA_WHISPER_MODEL", "small.en")
    threshold = voice._listener.threshold
    followup_sec = voice._followup_window_sec

    if fallback_from:
        wake_display = f"{wake_name} (FALLBACK from missing {fallback_from!r})"
    else:
        wake_display = wake_name

    print(
        fallback_from
    )
    print(
        f"[voice] mode=voice  wake={wake_display}  whisper={whisper_name}  "
        f"threshold={threshold:.2f}  followup={followup_sec:g}s"
    )
    print(f"[voice] say '{wake_name}' to speak; Ctrl-C to quit.")
    return voice
