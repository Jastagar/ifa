from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import httpx

from ifa.services import prompts

APP_PRIORITY = {
    "com.whatsapp": "high",
    "com.whatsapp.w4b": "high",
    "org.telegram.messenger": "high",
    "com.google.android.apps.messaging": "high",
    "com.google.android.dialer": "high",
    "com.samsung.android.dialer": "high",
    "com.google.android.gm": "normal",
    "com.Slack": "normal",
}
DEFAULT_APP_PRIORITY = "low"
MAX_NOTE_CHARS = 300
PRIORITIES = ("high", "normal", "low")

URGENT_ANNOUNCEMENT = "Sir, there is an urgent message for you."

SETTING_LIMITS = {
    "settleSec": (1, 60),
    "ttlSec": (10, 3600),
    "maxRecent": (1, 50),
    "maxDiscarded": (1, 500),
    "listenSec": (2, 30),
}


def parse_settings(raw: dict) -> dict:
    values = {}
    for key, (low, high) in SETTING_LIMITS.items():
        if raw.get(key) is None:
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        values[key] = int(value) if key.startswith("max") else value
    return values


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass
class Message:
    text: str
    received_at: float


@dataclass
class Burst:
    app: str
    package: str
    sender: str
    priority: str
    messages: list[Message] = field(default_factory=list)
    urgent: bool = False
    classified_count: int = 0
    announced: str | None = None
    read: bool = False
    note: str = ""

    @property
    def last_at(self) -> float:
        return self.messages[-1].received_at


class NotificationInbox:
    def __init__(
        self,
        max_recent: int | None = None,
        ttl_sec: float | None = None,
        settle_sec: float | None = None,
        max_discarded: int | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_recent = max_recent or int(_env_float("IFA_NOTIF_MAX_RECENT", 7))
        self.ttl_sec = ttl_sec if ttl_sec is not None else _env_float("IFA_NOTIF_TTL_SEC", 90)
        self.settle_sec = settle_sec if settle_sec is not None else _env_float("IFA_NOTIF_SETTLE_SEC", 4)
        max_discarded = max_discarded or int(_env_float("IFA_NOTIF_MAX_DISCARDED", 50))
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: list[Burst] = []
        self._discarded: deque[Burst] = deque(maxlen=max_discarded)

    def add(self, app: str, package: str, sender: str, text: str, priority: str, note: str = "") -> Burst | None:
        now = self._clock()
        with self._lock:
            self._flush_locked(now)
            burst = next(
                (b for b in self._recent if b.package == package and b.sender == sender),
                None,
            )
            if burst is None:
                burst = Burst(app=app, package=package, sender=sender, priority=priority)
                self._recent.append(burst)
            if note:
                burst.note = note
            elif any(m.text == text for m in burst.messages):
                return None
            burst.messages.append(Message(text=text, received_at=now))
            self._enforce_max_locked()
            return burst

    def configure(
        self,
        settle_sec: float | None = None,
        ttl_sec: float | None = None,
        max_recent: int | None = None,
        max_discarded: int | None = None,
    ) -> None:
        with self._lock:
            if settle_sec is not None:
                self.settle_sec = settle_sec
            if ttl_sec is not None:
                self.ttl_sec = ttl_sec
            if max_recent is not None:
                self.max_recent = max_recent
            if max_discarded is not None and max_discarded != self._discarded.maxlen:
                self._discarded = deque(self._discarded, maxlen=max_discarded)
            self._flush_locked(self._clock())
            self._enforce_max_locked()

    def settings(self) -> dict:
        with self._lock:
            return {"settleSec": self.settle_sec, "ttlSec": self.ttl_sec,
                    "maxRecent": self.max_recent, "maxDiscarded": self._discarded.maxlen}

    def settled(self) -> list[Burst]:
        now = self._clock()
        with self._lock:
            self._flush_locked(now)
            return [
                b for b in self._recent
                if now - b.last_at >= self.settle_sec
                and len(b.messages) > b.classified_count
            ]

    def take_unread(self) -> list[Burst]:
        with self._lock:
            self._flush_locked(self._clock())
            taken, self._recent = self._recent, []
            for burst in taken:
                burst.read = True
                self._discarded.append(burst)
            return taken

    def older(self, limit: int = 10) -> list[Burst]:
        with self._lock:
            self._flush_locked(self._clock())
            return list(reversed(self._discarded))[:limit]

    def unread_count(self) -> int:
        with self._lock:
            self._flush_locked(self._clock())
            return len(self._recent)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked(self._clock())

    def _flush_locked(self, now: float) -> None:
        expired = [b for b in self._recent if now - b.last_at > self.ttl_sec]
        for burst in expired:
            self._recent.remove(burst)
            self._discarded.append(burst)

    def _enforce_max_locked(self) -> None:
        while len(self._recent) > self.max_recent:
            oldest = min(self._recent, key=lambda b: b.last_at)
            self._recent.remove(oldest)
            self._discarded.append(oldest)


URGENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "urgent": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["urgent", "reason"],
}

URGENCY_PROMPT = prompts.URGENCY


_structured_output_supported = True


def _parse_verdict(content: str) -> dict:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in reply: {content[:80]!r}")
    return json.loads(match.group(0))


def classify_urgency(burst: Burst, chat_fn: Callable[..., dict], model: str) -> bool:
    global _structured_output_supported
    lines = "\n".join(f"- {m.text}" for m in burst.messages)
    data = (
        f"App: {burst.app} (user's priority for this app: {burst.priority})\n"
        f"Sender: {burst.sender}\n"
        f"Messages ({len(burst.messages)}, oldest first):\n{lines}"
    )
    note = (
        f"Sir's own note about {burst.app} notifications, from his settings (trust it): {burst.note}\n\n"
        if burst.note else ""
    )
    messages = [
        {"role": "system", "content": prompts.get("urgency")},
        {"role": "user", "content": f"{note}<notification>\n{data}\n</notification>"},
    ]
    try:
        response = None
        if _structured_output_supported:
            try:
                response = chat_fn(model=model, messages=messages, think=False, format=URGENCY_SCHEMA)
            except httpx.HTTPStatusError as exc:
                _structured_output_supported = False
                print(f"[notify] structured output unavailable ({exc.response.status_code}); "
                      "using prompt-only JSON from now on")
        if response is None:
            response = chat_fn(model=model, messages=messages, think=False)
        verdict = _parse_verdict(response["message"]["content"])
        print(f"[notify] {burst.sender}: urgent={verdict.get('urgent')} ({verdict.get('reason')})")
        return verdict.get("urgent") is True
    except Exception as exc:
        print(f"[notify] ⚠ urgency check FAILED ({exc}); treating as not urgent")
        return False


SUMMARY_PROMPT = prompts.SUMMARY


def template_summary(bursts: list[Burst]) -> str:
    lines = []
    for b in bursts:
        latest = next((m.text for m in reversed(b.messages) if m.text), "")
        prefix = "Urgent: " if b.urgent else ""
        quote = f': "{latest}"' if latest else "."
        lines.append(f"{prefix}{b.sender} sent you a {b.app} message{quote}")
    return "\n".join(lines)


def summarize_bursts(bursts: list[Burst], chat_fn: Callable[..., dict], model: str) -> str:
    try:
        response = chat_fn(
            model=model,
            messages=[
                {"role": "system", "content": prompts.get("summary")},
                {"role": "user", "content": format_bursts(bursts)},
            ],
            think=False,
        )
        summary = response["message"]["content"].strip()
        if not summary:
            raise ValueError("empty summary")
        return summary
    except Exception as exc:
        print(f"[notify] ⚠ summary FAILED ({exc}); using the template")
        return template_summary(bursts)


class NotificationCenter:
    def __init__(
        self,
        inbox: NotificationInbox,
        classify: Callable[[Burst], bool],
        summarize: Callable[[list[Burst]], str] = template_summary,
        on_announce: Callable[[str], None] | None = None,
        poll_sec: float = 0.5,
    ) -> None:
        self.inbox = inbox
        self._classify = classify
        self.summarize = summarize
        self.on_announce = on_announce
        self._poll_sec = poll_sec
        self._lock = threading.Lock()
        self._pending: str | None = None
        self.announcement_ready = threading.Event()

    def receive(self, payload: dict) -> Burst | None:
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            raise ValueError("`context` must be an object")
        package = str(context.get("packageName") or "unknown")
        app = str(context.get("appName") or package)
        sender = str(context.get("title") or "Unknown sender")
        text = str(context.get("body") or context.get("text") or "").strip()
        priority = str(context.get("priority") or payload.get("priority") or "").lower()
        if priority not in PRIORITIES:
            priority = APP_PRIORITY.get(package, DEFAULT_APP_PRIORITY)
        note = " ".join(str(context.get("appNote") or "").split())[:MAX_NOTE_CHARS]

        burst = self.inbox.add(app=app, package=package, sender=sender, text=text, priority=priority, note=note)
        if burst is not None:
            print(f"[notify] {app} · {sender} ({priority}): {len(burst.messages)} message(s)")
        return burst

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._run, name="ifa-notifications", daemon=True)
        thread.start()
        return thread

    def process_settled(self) -> None:
        for burst in self.inbox.settled():
            burst.classified_count = len(burst.messages)
            if burst.priority == "low":
                continue
            burst.urgent = self._classify(burst)
            level = "urgent" if burst.urgent else "normal"
            if burst.announced is None or (level == "urgent" and burst.announced != "urgent"):
                burst.announced = level
                self._queue_announcement(level)

    def take_announcement(self) -> str | None:
        with self._lock:
            level, self._pending = self._pending, None
            self.announcement_ready.clear()
            return level

    def _queue_announcement(self, level: str) -> None:
        if self.on_announce is not None:
            self.on_announce(level)
            return
        self.hold_announcement(level)

    def hold_announcement(self, level: str) -> None:
        with self._lock:
            if self._pending != "urgent":
                self._pending = level
            self.announcement_ready.set()

    def _run(self) -> None:
        while True:
            try:
                self.inbox.flush()
                self.process_settled()
            except Exception as exc:
                print(f"[notify] worker error: {exc}")
            time.sleep(self._poll_sec)


def format_bursts(bursts: list[Burst], now: float | None = None) -> str:
    if not bursts:
        return "No notifications."
    now = now if now is not None else time.time()
    blocks = []
    for b in bursts:
        ago = int(now - b.last_at)
        when = f"{ago}s ago" if ago < 120 else f"{ago // 60} min ago"
        flag = ", URGENT" if b.urgent else ""
        texts = "\n".join(f"  - {m.text or '(no text)'}" for m in b.messages)
        blocks.append(f"{b.app} · from {b.sender} · {when}{flag}\n{texts}")
    return "\n\n".join(blocks)
