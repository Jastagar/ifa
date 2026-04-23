import os
import queue
import tempfile
import threading

import pyttsx3
import sounddevice as sd
import soundfile as sf
from rich.console import Console

_console = Console()


class TTSService:
    """Cross-platform text-to-speech.

    Synthesis uses `pyttsx3` (OS-native voices, offline). Playback uses
    `sounddevice` on the system default output.

    pyttsx3 on macOS uses Cocoa's NSRunLoop, which is not safe to drive from
    background threads. `speak()` called from a non-main thread posts the
    request to an internal queue and blocks; the main thread drains the queue
    via `drain_queue()` on each orchestrator loop iteration.
    """

    def __init__(self):
        self._main_thread_id = threading.get_ident()
        self._queue: "queue.Queue" = queue.Queue()
        self._speak_lock = threading.Lock()

    def speak(self, text: str) -> None:
        if not text:
            return

        if threading.get_ident() == self._main_thread_id:
            self._speak_on_main(text)
            return

        done = threading.Event()
        self._queue.put((text, done))
        done.wait()

    def drain_queue(self) -> None:
        """Run every pending speak request on the main thread. Non-blocking
        if the queue is empty."""
        while True:
            try:
                text, done = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._speak_on_main(text)
            finally:
                done.set()

    def _speak_on_main(self, text: str) -> None:
        with self._speak_lock:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="ifa_tts_")
                os.close(fd)

                engine = pyttsx3.init()
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()

                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    _console.print("[yellow]TTS produced no audio; skipping playback.[/yellow]")
                    return

                data, samplerate = sf.read(tmp_path, dtype="float32")

                try:
                    sd.play(data, samplerate, device=None)
                    sd.wait()
                except sd.PortAudioError as exc:
                    _console.print(f"[yellow]TTS playback failed: {exc}[/yellow]")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
