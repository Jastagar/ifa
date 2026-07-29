
import os
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import torch
import contextlib
import io
os.environ["TQDM_DISABLE"] = "1"

from chatterbox.tts_turbo import ChatterboxTurboTTS
# from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from blingfire import text_to_sentences
from rich.console import Console

_console = Console()

class TTSService:
    def __init__(self):

        self._lock = threading.Lock()
        self._active_count = 0
        self._mute_until = 0.0

        self._cooldown_sec = (
            float(os.environ.get("IFA_TTS_COOLDOWN_MS", "500")) / 1000.0
        )

        self._stop_event = threading.Event()

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        
        _console.print("PROCESSOR TYPE :",torch.cuda.is_available())

        _console.print(
            f"Loading Chatterbox Turbo on {self.device}"
        )

        self.model = ChatterboxTurboTTS.from_pretrained(
            device=self.device
        )
        # self.model = ChatterboxMultilingualTTS.from_pretrained(
        #     device=self.device
        # )

        self.voice_path = os.environ.get(
            "IFA_VOICE_SAMPLE",
            "ifa/audios/voice.wav"
        )

        # Text generation and audio playback run independently. This lets the
        # GPU prepare the next sentence/phrase while the speaker is playing
        # the previous one.
        self._text_queue = queue.Queue()
        # Buffer enough generated audio to stay ahead of playback for longer
        # answers without unbounded memory growth.
        self._audio_queue = queue.Queue(maxsize=64)
        self._producer = threading.Thread(target=self._produce_audio, daemon=True)
        self._consumer = threading.Thread(target=self._play_audio, daemon=True)
        self._producer.start()
        self._consumer.start()

    @property
    def is_speaking(self):

        with self._lock:
            return (
                self._active_count > 0
                or time.monotonic() < self._mute_until
            )

    def stop(self):
        self._stop_event.set()

    def enqueue(self, text: str) -> threading.Event | None:
        """Queue speech without waiting for earlier audio to finish."""
        if not text:
            return None

        with self._lock:
            self._active_count += 1

        self._stop_event.clear()
        completed = threading.Event()
        self._text_queue.put((text.strip(), completed))
        return completed

    def speak(self, text: str):
        """Queue speech and wait for its audio to finish (legacy API)."""
        completed = self.enqueue(text)
        if completed:
            completed.wait()

    def _produce_audio(self) -> None:
        while True:
            text, completed = self._text_queue.get()
            try:
                for sentence in self._split_sentences(text):
                    if self._stop_event.is_set():
                        break

                    with contextlib.redirect_stdout(io.StringIO()):
                        with contextlib.redirect_stderr(io.StringIO()):
                            wav = self.model.generate(
                                sentence,
                                audio_prompt_path=self.voice_path
                            )

                    audio = (
                        wav.squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    self._audio_queue.put((audio, None))

            except Exception as exc:
                _console.print(f"Producer error: {exc}")

            finally:
                self._audio_queue.put((None, completed))
                self._text_queue.task_done()

    def _play_audio(self) -> None:
        stream = None
        while True:
            audio, completed = self._audio_queue.get()
            try:
                if audio is not None:
                    if stream is None:
                        stream = sd.OutputStream(
                            samplerate=self.model.sr,
                            channels=1,
                            dtype="float32",
                            blocksize=2048,
                        )
                        stream.start()
                    stream.write(audio)
                elif completed is not None:
                    with self._lock:
                        self._active_count -= 1
                        self._mute_until = max(
                            self._mute_until,
                            time.monotonic() + self._cooldown_sec,
                        )
                    completed.set()
            finally:
                self._audio_queue.task_done()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        lines = [s for s in text_to_sentences(text).splitlines() if s]
        return lines 