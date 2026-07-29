"""Speech-to-text via Whisper-Hindi2Hinglish-Swift.

Primary entry point for Stage 2 voice mode is ``transcribe_array`` —
Unit 3's capture step produces a float32 numpy array at 16 kHz, and
this module runs it through Whisper and returns the transcribed text.

Model is loaded lazily the first time a transcribe function is called,
then reused across turns.

Uses:
    Oriserve/Whisper-Hindi2Hinglish-Swift

Optimized for Hindi + Hinglish conversational speech.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


_model: Optional[object] = None
_processor: Optional[object] = None


def _get_model():
    """Load Whisper model lazily."""

    global _model, _processor

    if _model is None:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        model_name = os.environ.get(
            "IFA_WHISPER_MODEL",
            "Oriserve/Whisper-Hindi2Hinglish-Swift",
        )

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        dtype = (
            torch.float16
            if device == "cuda"
            else torch.float32
        )

        print(
            f"[whisper] loading {model_name} "
            f"on {device}"
        )

        _processor = AutoProcessor.from_pretrained(
            model_name
        )

        _model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        _model.to(device)
        _model.eval()

        print(
            f"[whisper] loaded {model_name}"
        )

    return _model, _processor


def transcribe_array(
    audio: np.ndarray,
    *,
    language: str = "hi"
) -> str:
    """
    Transcribe float32 numpy array at 16kHz mono.

    Keeps the same interface as faster-whisper.
    """

    if audio is None or len(audio) == 0:
        return ""

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    try:
        import torch

        model, processor = _get_model()

        device = next(model.parameters()).device

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(
                device=device,
                dtype=next(model.parameters()).dtype
                if v.dtype.is_floating_point
                else v.dtype,
            )
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                language=language,
                task="transcribe",
            )

        text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0]

        return text.strip()

    except Exception as exc:
        print(f"[whisper] transcription failed: {exc}")
        return ""


def transcribe(audio_path: str) -> str:
    """
    File-path compatibility wrapper.
    """

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32",
        )

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resampling if required
        if sample_rate != 16000:
            import librosa

            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=16000,
            )

        return transcribe_array(audio)

    except Exception as exc:
        print(f"[whisper] file transcription failed: {exc}")
        return ""