from __future__ import annotations

import os
import random
import threading

import numpy as np

GENERIC = [
    "One sec.",
    "Just a sec.",
    "One moment.",
    "Give me a second.",
]

BY_TOOL = {
    "get_time": "Checking the time.",
    "set_reminder": "Setting that up.",
    "remember": "Noting that down.",
    "analyze_screen": "Let me take a look.",
    "spotify_control": "On it.",
    "open_app": "Opening that.",
    "open_website": "Opening that.",
    "launch_game": "Starting it up.",
    "switch_mode": "Switching modes.",
    "whats_app": "Sending that over.",
    "call_n8n_workflow": "Running that workflow.",
    "read_notifications": "Let me check.",
}


def generic_phrases() -> list[str]:
    raw = os.environ.get("IFA_FILLER_PHRASES", "")
    return [p.strip() for p in raw.split("|") if p.strip()] or list(GENERIC)


class Fillers:
    def __init__(self, synthesize, generic=GENERIC, by_tool=BY_TOOL, extra=(), rng=None) -> None:
        self._synthesize = synthesize
        self._generic = list(generic)
        self._by_tool = dict(by_tool)
        self._extra = list(extra)
        self._rng = rng or random.Random()
        self._clips: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._last_generic: str | None = None

    def prepare_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self._prepare_all, name="ifa-fillers", daemon=True)
        thread.start()
        return thread

    def _prepare_all(self) -> None:
        for phrase in [*self._extra, *self._generic, *self._by_tool.values()]:
            try:
                audio = self._synthesize(phrase)
            except Exception as exc:
                print(f"[fillers] could not render {phrase!r}: {exc}")
                continue
            with self._lock:
                self._clips[phrase] = audio

    def clip(self, phrase: str) -> np.ndarray | None:
        with self._lock:
            return self._clips.get(phrase)

    def choose(self, tool_names: list[str] | None = None) -> tuple[str, np.ndarray] | None:
        with self._lock:
            for name in tool_names or []:
                phrase = self._by_tool.get(name)
                if phrase in self._clips:
                    return phrase, self._clips[phrase]

            ready = [p for p in self._generic if p in self._clips]
            if len(ready) > 1 and self._last_generic in ready:
                ready.remove(self._last_generic)
            if not ready:
                return None
            phrase = self._rng.choice(ready)
            self._last_generic = phrase
            return phrase, self._clips[phrase]
