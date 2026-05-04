import base64
import os
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
import queue
from kokoro import KPipeline
import sounddevice as sd
from rich.console import Console

_console = Console()


class TTSService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_count = 0
        self._mute_until = 0.0
        self._cooldown_sec = float(os.environ.get("IFA_TTS_COOLDOWN_MS", "500")) / 1000.0

        # 🔥 Kokoro model (initialize once)
        self.kokoro_model = KPipeline(lang_code="a")

        # 🔥 interrupt support
        self._stop_event = threading.Event()

    # ----------------------------
    # STATE
    # ----------------------------
    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._active_count > 0 or time.monotonic() < self._mute_until

    def stop(self):
        """Interrupt ongoing speech"""
        self._stop_event.set()

    # ----------------------------
    # MAIN ENTRY
    # ----------------------------
    def speak(self, text: str) -> None:
        if not text:
            return

        with self._lock:
            self._active_count += 1

        self._stop_event.clear()

        try:
            # ✅ Try Kokoro first
            try:
                self._speak_kokoro_stream(text)
                return
            except Exception as e:
                _console.print(f"[yellow]Kokoro failed, falling back: {e}[/yellow]")

            # ✅ Fallback
            if sys.platform == "darwin":
                self._speak_macos(text)
            elif sys.platform == "win32":
                self._speak_windows(text)
            else:
                subprocess.run(["espeak", "--", text], check=False)

        finally:
            with self._lock:
                self._active_count -= 1
                self._mute_until = max(
                    self._mute_until, time.monotonic() + self._cooldown_sec
                )

    # ----------------------------
    # KOKORO STREAMING
    # ----------------------------
    def _speak_kokoro_stream(self, text: str):
        audio_queue = queue.Queue(maxsize=8)
        stop_signal = object()

        def producer():
            try:
                # Kokoro yields (text, phonemes, audio)
                for _, _, audio in self.kokoro_model(text, voice="af_heart"):
                    if self._stop_event.is_set():
                        break
                    audio_queue.put(np.array(audio, dtype=np.float32))
            except Exception as e:
                _console.print(f"[red]Producer error: {e}[/red]")
            finally:
                audio_queue.put(stop_signal)

        def consumer(sample_rate=22050):
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
            )
            stream.start()

            try:
                while not self._stop_event.is_set():
                    try:
                        chunk = audio_queue.get(timeout=1)
                    except queue.Empty:
                        continue

                    if chunk is stop_signal:
                        break

                    stream.write(chunk)
            finally:
                stream.stop()
                stream.close()

        t1 = threading.Thread(target=producer, daemon=True)
        t2 = threading.Thread(target=consumer, daemon=True)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

    # ----------------------------
    # FALLBACKS
    # ----------------------------
    def _speak_macos(self, text: str) -> None:
        fd, aiff_path = tempfile.mkstemp(suffix=".aiff", prefix="ifa_tts_")
        os.close(fd)
        try:
            subprocess.run(["say", "-o", aiff_path, "--", text], check=False)
            subprocess.run(["afplay", aiff_path], check=False)
        finally:
            try:
                os.unlink(aiff_path)
            except OSError:
                pass

    def _speak_windows(self, text: str) -> None:
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            '$s.Speak([System.Environment]::GetEnvironmentVariable("IFA_TTS_TEXT"))'
        )

        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        env = {**os.environ, "IFA_TTS_TEXT": text}

        subprocess.run(
            ["powershell", "-NoProfile", "-EncodedCommand", encoded],
            env=env,
            check=False,
        )