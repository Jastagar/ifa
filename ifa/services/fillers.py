"""Pre-rendered filler clips ("One moment.", "Checking the time.") that cover silence.

A filler only has to sound natural and arrive instantly. It doesn't need
an LLM: a lookup table can't hallucinate ("Reminder set!" before anything
was set) and costs nothing at runtime. Each phrase is synthesized once, in
Ifa's own voice, on a background thread at startup; playing one later
skips synthesis entirely.

Phrases describe work *in progress*, never a result — the action may still
fail, and Ifa shouldn't have claimed it worked.
"""
from __future__ import annotations

import random
import threading

import numpy as np

GENERIC = [
    "One moment.",
    "Let me see.",
    "Hmm, let me think.",
    "Okay, give me a second.",
    "Sure, one sec.",
]

# Played as soon as the model calls a tool: we know a second LLM round is
# coming, so the answer will take a while.
BY_TOOL = {
    "get_time": "Checking the time.",
    "set_reminder": "Setting that up.",
    "remember": "Noting that down.",
    "analyze_screen": "Let me take a look.",
    "spotify_control": "On it.",
    "open_app": "Opening that.",
    "launch_game": "Starting it up.",
    "switch_mode": "Switching modes.",
    "whats_app": "Sending that over.",
    "call_n8n_workflow": "Running that workflow.",
    "read_notifications": "Let me check.",
}


class Fillers:
    def __init__(self, synthesize, generic=GENERIC, by_tool=BY_TOOL, extra=(), rng=None) -> None:
        self._synthesize = synthesize
        self._generic = list(generic)
        self._by_tool = dict(by_tool)
        self._extra = list(extra)  # other fixed lines, e.g. the urgent-notification announcement
        self._rng = rng or random.Random()
        self._clips: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._last_generic: str | None = None

    def prepare_in_background(self) -> threading.Thread:
        """Render every phrase on a daemon thread. Fillers become usable as they finish."""
        thread = threading.Thread(target=self._prepare_all, name="ifa-fillers", daemon=True)
        thread.start()
        return thread

    def _prepare_all(self) -> None:
        # Extra lines first (announcements), then generic fillers (they cover
        # every slow turn), then tool phrases (they cover only some).
        for phrase in [*self._extra, *self._generic, *self._by_tool.values()]:
            try:
                audio = self._synthesize(phrase)
            except Exception as exc:
                print(f"[fillers] could not render {phrase!r}: {exc}")
                continue
            with self._lock:
                self._clips[phrase] = audio

    def clip(self, phrase: str) -> np.ndarray | None:
        """The rendered audio for a fixed phrase, or None if it isn't ready."""
        with self._lock:
            return self._clips.get(phrase)

    def choose(self, tool_names: list[str] | None = None) -> tuple[str, np.ndarray] | None:
        """Pick a ready clip: tool-specific if one matches, else a generic one.

        Returns None if nothing is rendered yet — better silence than a
        filler that has to be synthesized on the spot.
        """
        with self._lock:
            for name in tool_names or []:
                phrase = self._by_tool.get(name)
                if phrase in self._clips:
                    return phrase, self._clips[phrase]

            ready = [p for p in self._generic if p in self._clips]
            if len(ready) > 1 and self._last_generic in ready:
                ready.remove(self._last_generic)  # don't say the same thing twice in a row
            if not ready:
                return None
            phrase = self._rng.choice(ready)
            self._last_generic = phrase
            return phrase, self._clips[phrase]
