import os
import re
import threading
import queue
import time

import numpy as np
import sounddevice as sd
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS
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

        self.voice_path = os.environ.get(
            "IFA_VOICE_SAMPLE",
            "ifa/audios/voice.wav"
        )

    @property
    def is_speaking(self):

        with self._lock:
            return (
                self._active_count > 0
                or time.monotonic() < self._mute_until
            )

    def stop(self):
        self._stop_event.set()

    def speak(self, text: str):

        if not text:
            return

        with self._lock:
            self._active_count += 1

        self._stop_event.clear()

        try:
            self._speak_stream(text)

        finally:

            with self._lock:
                self._active_count -= 1

                self._mute_until = max(
                    self._mute_until,
                    time.monotonic() + self._cooldown_sec,
                )

    def _speak_stream(self, text: str):

        audio_queue = queue.Queue(maxsize=8)

        stop_signal = object()

        chunks = self._split_sentences(text)

        def producer():

            try:
                for chunk in chunks:

                    if self._stop_event.is_set():
                        break

                    wav = self.model.generate(
                        chunk,
                        audio_prompt_path=self.voice_path,
                    )

                    audio = (
                        wav.squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    audio_queue.put(audio)

            except Exception as e:
                _console.print(
                    f"Producer error: {e}"
                )

            finally:
                audio_queue.put(stop_signal)

        def consumer():

            stream = sd.OutputStream(
                samplerate=self.model.sr,
                channels=1,
                dtype="float32",
                blocksize=2048,
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

        t1 = threading.Thread(
            target=producer,
            daemon=True
        )

        t2 = threading.Thread(
            target=consumer,
            daemon=True
        )

        t1.start()
        t2.start()

        t1.join()
        t2.join()

    def _split_sentences(self, text: str):

        text = text.strip()

        if not text:
            return []

        return [
            s.strip()
            for s in re.split(
                r"(?<=[.!?])\s+",
                text
            )
            if s.strip()
        ]