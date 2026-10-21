from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

from ifa.utils.devices import resolve_device


DEFAULT_MODEL = "Oriserve/Whisper-Hindi2Hinglish-Apex"

_model: Optional[object] = None
_processor: Optional[object] = None
_prompt_ids: Optional[object] = None
_lock = threading.RLock()


def _get_model():
    with _lock:
        return _load_model()


def _load_model():
    global _model, _processor

    if _model is None:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        model_name = os.environ.get("IFA_WHISPER_MODEL", "").strip() or DEFAULT_MODEL

        device = resolve_device("IFA_WHISPER_DEVICE")

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


def _prompt_kwargs(processor, device) -> dict:
    global _prompt_ids
    prompt = os.environ.get("IFA_WHISPER_PROMPT", "").strip()
    if not prompt:
        return {}
    if _prompt_ids is None:
        _prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt")
    return {"prompt_ids": _prompt_ids.to(device)}


def transcribe_array(
    audio: np.ndarray,
    *,
    language: Optional[str] = None,
) -> str:
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

        with _lock, torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                language=language or os.environ.get("IFA_WHISPER_LANGUAGE", "en"),
                **_prompt_kwargs(processor, device),
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