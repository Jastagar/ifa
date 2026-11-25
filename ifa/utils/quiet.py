"""Keep the console readable without hiding Ifa's own lines.

Libraries print progress bars and chatter. A redirect around TTS used to hide them, but redirects are
process-wide and swallowed every other thread's output too. Instead: switch the bars off through the
libraries' own environment switches (read when tqdm is first imported, so this must run first), and drop
only exact known noise lines.
"""
from __future__ import annotations

import os
import sys

NOISE_LINES = {"S3 Token -> Mel Inference..."}  # Chatterbox prints this on every sentence


class _DropNoise:
    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        if text.strip() in NOISE_LINES:
            return len(text)
        return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if not isinstance(sys.stdout, _DropNoise):
        sys.stdout = _DropNoise(sys.stdout)
