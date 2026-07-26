import os
import re
import threading
import queue
import time

import numpy as np
import sounddevice as sd
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS
# from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from rich.console import Console
from blingfire import text_to_sentences

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

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
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        
        _console.print("PROCESSOR TYPE :",torch.cuda.is_available())

        _console.print(
            f"Loading Chatterbox Turbo on {self.device}"
        )

        _console.print(
            f"Loading Indic Parler TTS on {self.device}"
        )

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts"
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/indic-parler-tts"
        )

        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )

        self.sample_rate = self.model.config.sampling_rate
        # self.model = ChatterboxMultilingualTTS.from_pretrained(
        #     device=self.device
        # )

        self.voice_description = "Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise. She speaks Hinglish"
        desc = self.description_tokenizer(
            self.voice_description,
            return_tensors="pt"
        )

        self.description_inputs = {
            k: v.to(self.device)
            for k, v in desc.items()
        }

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

                    prompt_inputs = self.tokenizer(
                        sentence,
                        return_tensors="pt",
                    )

                    prompt_inputs = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in prompt_inputs.items()
                    }

                    with torch.inference_mode():

                        generation = self.model.generate(
                            input_ids=self.description_inputs["input_ids"],
                            attention_mask=self.description_inputs["attention_mask"],
                            prompt_input_ids=prompt_inputs["input_ids"],
                            prompt_attention_mask=prompt_inputs["attention_mask"],
                        )

                    if isinstance(generation, tuple):
                        generation = generation[0]

                    if isinstance(generation, dict):
                        generation = generation["audio"]

                    audio = (
                        generation
                        .detach()
                        .float()
                        .cpu()
                        .squeeze()
                        .numpy()
                        .astype(np.float32)
                    )

                    if self._stop_event.is_set():
                        break

                    self._audio_queue.put((audio, None))

            except Exception as exc:
                _console.print(f"[red]Producer error:[/red] {exc}")

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
                            samplerate=self.sample_rate,
                            channels=1,
                            dtype="float32",
                            blocksize=4096,
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
        return [s for s in text_to_sentences(text) if s]

