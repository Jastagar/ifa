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

import perth

if perth.PerthImplicitWatermarker is None:
    print("[tts] Chatterbox's audio watermark couldn't load (setuptools too new? run Install packages); "
          "speaking without it")
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

from chatterbox.tts_turbo import ChatterboxTurboTTS
from rich.console import Console

from ifa.utils import latency
from ifa.utils.devices import resolve_device
from ifa.utils.sentences import split_sentences

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

        # On a Mac, Chatterbox ran no faster on the Apple GPU than on the CPU (both ~1.0-1.3x
        # real time on an M4 Pro), and the GPU is busy with Ollama, so auto picks the CPU there.
        self.device = resolve_device("IFA_TTS_DEVICE", auto_mps=False)


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
        self._model_lock = threading.Lock()

        self._text_queue = queue.Queue()
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

    def enqueue(self, text: str, sink=None) -> threading.Event | None:
        if not text:
            return None

        if sink is None:
            with self._lock:
                self._active_count += 1

        self._stop_event.clear()
        completed = threading.Event()
        self._text_queue.put((text.strip(), completed, sink))
        return completed

    def speak(self, text: str):
        completed = self.enqueue(text)
        if completed:
            completed.wait()

    def synthesize(self, text: str) -> np.ndarray:
        return self._generate(text.strip())

    def play_clip(self, audio: np.ndarray, label: str, sink=None) -> threading.Event:
        if sink is None:
            with self._lock:
                self._active_count += 1
        completed = threading.Event()
        self._audio_queue.put((audio, None, label, sink))
        self._audio_queue.put((None, completed, label, sink))
        return completed

    @property
    def sample_rate(self) -> int:
        return int(self.model.sr)

    def _generate(self, sentence: str) -> np.ndarray:
        with self._model_lock:
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    wav = self.model.generate(
                        sentence,
                        audio_prompt_path=self.voice_path
                    )
        return (
            wav.squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _produce_audio(self) -> None:
        while True:
            text, completed, sink = self._text_queue.get()
            try:
                for sentence in self._split_sentences(text):
                    if self._stop_event.is_set():
                        break

                    synth_started = time.perf_counter()
                    audio = self._generate(sentence)
                    latency.current().record_tts(
                        chars=len(sentence),
                        synth_ms=round((time.perf_counter() - synth_started) * 1000),
                        audio_ms=round(len(audio) / self.model.sr * 1000),
                    )

                    self._audio_queue.put((audio, None, text, sink))

            except Exception as exc:
                _console.print(f"Producer error: {exc}")

            finally:
                self._audio_queue.put((None, completed, text, sink))
                self._text_queue.task_done()

    def _play_audio(self) -> None:
        stream = None
        while True:
            audio, completed, source_text, sink = self._audio_queue.get()
            try:
                if sink is not None:
                    if audio is not None:
                        latency.current().audio_started(source_text)
                        try:
                            sink(audio, self.model.sr)
                        except Exception as exc:
                            _console.print(f"Audio sink error: {exc}")
                    elif completed is not None:
                        completed.set()
                elif audio is not None:
                    if stream is None:
                        stream = sd.OutputStream(
                            samplerate=self.model.sr,
                            channels=1,
                            dtype="float32",
                            blocksize=2048,
                        )
                        stream.start()
                    latency.current().audio_started(source_text)
                    stream.write(audio)
                elif completed is not None:
                    with self._lock:
                        self._active_count -= 1
                        all_done = self._active_count == 0
                        self._mute_until = max(
                            self._mute_until,
                            time.monotonic() + self._cooldown_sec,
                        )
                    if all_done:
                        latency.current().mark("audio_done", last=True)
                    completed.set()
            finally:
                self._audio_queue.task_done()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return split_sentences(text) 