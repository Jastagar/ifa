from __future__ import annotations

import re
import threading

import pysbd

# pySBD (Pragmatic Sentence Boundary Detection): rule-based, pure Python, offline. It knows abbreviations
# ("Dr.", "e.g.", "U.S.", "p.m."), decimals ("572.60"), URLs ("youtube.com") and numbered lists.
_segmenter = pysbd.Segmenter(language="en", clean=False)
_lock = threading.Lock()  # a Segmenter keeps state while it works, so one call at a time

LONG_SENTENCE_WORDS = 20
_CLAUSE_BREAK = re.compile(r"[,;:]\s|\s[–—-]\s")


def _segment(text: str) -> list[str]:
    with _lock:
        return [s.strip() for s in _segmenter.segment(text) if s.strip()]


def split_sentences(text: str) -> list[str]:
    """Split finished text into sentences for speech."""
    return [s for paragraph in re.split(r"\n+", text) for s in _segment(paragraph)]


def take_complete(buffer: str) -> tuple[list[str], str]:
    """For streamed text: the sentences that are surely finished, and the rest to keep buffering.

    The last sentence pySBD finds may still be growing ("572." before ".60" arrives), so it waits for
    more text; the caller speaks whatever is left when the stream ends. A sentence that runs long is cut
    at its last comma or semicolon, so speech can start before the whole sentence is written.
    """
    parts = [s for paragraph in re.split(r"\n+", buffer) for s in _segment(paragraph)]
    if buffer.endswith("\n") and parts:
        return parts, ""
    if len(parts) < 2:
        return _cut_long(buffer)
    complete, tail = parts[:-1], buffer[buffer.rfind(parts[-1]):]
    more, tail = _cut_long(tail)
    return complete + more, tail


def _cut_long(text: str) -> tuple[list[str], str]:
    if len(text.split()) < LONG_SENTENCE_WORDS:
        return [], text
    breaks = [m.end() for m in _CLAUSE_BREAK.finditer(text) if len(text[:m.end()].split()) >= 6]
    if not breaks:
        return [], text
    return [text[:breaks[-1]].strip()], text[breaks[-1]:].lstrip()
