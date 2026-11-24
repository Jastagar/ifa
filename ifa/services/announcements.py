"""Spoken announcements from outside Ifa, such as Claude Code sessions finishing (see scripts/claude_hook.py)."""
from __future__ import annotations

from typing import Callable

from ifa.services import prompts

MAX_INPUT_CHARS = 4000
KINDS = ("done", "needs_input")


def parse(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("the body must be a JSON object")
    kind = str(payload.get("kind") or "")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    return {
        "kind": kind,
        "source": " ".join(str(payload.get("source") or "Claude").split())[:40],
        "project": " ".join(str(payload.get("project") or "").split())[:60],
        "text": str(payload.get("text") or "").strip()[:MAX_INPUT_CHARS],
    }


def spoken_text(event: dict, chat_fn: Callable[..., dict], model: str) -> str:
    """One sentence for Ifa to say. The local model condenses a finished session's last message."""
    where = f" in {event['project']}" if event["project"] else ""
    if event["kind"] == "needs_input":
        what = event["text"].rstrip(".") if event["text"] else "needs your input"
        return f"{event['source']}{where}: {what[0].lower() + what[1:] if what else what}."
    summary = ""
    if event["text"]:
        try:
            response = chat_fn(model=model, think=False, messages=[
                {"role": "system", "content": prompts.get("claude")},
                {"role": "user", "content": event["text"]},
            ])
            summary = " ".join(response["message"]["content"].split()).strip()
        except Exception as exc:
            print(f"[announce] couldn't summarise ({exc}); saying it's done")
    return f"{event['source']} finished{where}" + (f": {summary}" if summary else ".")
