#!/usr/bin/env python3
"""Claude Code hook: tell Ifa when a session finishes or needs you, so she says it out loud.

Register it for the Stop and Notification hooks in ~/.claude/settings.json (async, so Claude never waits):

    "hooks": {
      "Stop":         [{"hooks": [{"type": "command", "command": "python3 /path/to/ifa/scripts/claude_hook.py", "async": true, "timeout": 20}]}],
      "Notification": [{"hooks": [{"type": "command", "command": "python3 /path/to/ifa/scripts/claude_hook.py", "async": true, "timeout": 20}]}]
    }

Settings in ~/.ifa/claude_hook.json (or the IFA_URL / IFA_TOKEN environment variables):

    {"url": "http://192.168.1.15:8789", "token": "<the phone pairing code>", "skip_projects": []}

Standard library only. It always exits 0 and prints nothing, so it can never block or change Claude.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

CONFIG = Path.home() / ".ifa" / "claude_hook.json"
# Notification types worth saying out loud. idle_prompt is skipped: it repeats the Stop announcement a minute later.
NEEDS_YOU = {
    "permission_prompt": "needs your permission",
    "elicitation_dialog": "has a question for you",
    "agent_needs_input": "needs your input",
}


def event_for(hook: dict) -> dict | None:
    project = Path(hook.get("cwd") or "").name
    name = hook.get("hook_event_name")
    if name == "Stop":
        if hook.get("stop_hook_active"):
            return None
        return {"kind": "done", "source": "Claude", "project": project,
                "text": hook.get("last_assistant_message") or ""}
    if name == "Notification":
        what = NEEDS_YOU.get(hook.get("notification_type") or "")
        if what is None:
            return None
        return {"kind": "needs_input", "source": "Claude", "project": project, "text": what}
    return None


def main() -> int:
    try:
        hook = json.load(sys.stdin)
        config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
        url = os.environ.get("IFA_URL") or config.get("url") or "http://127.0.0.1:8789"
        token = os.environ.get("IFA_TOKEN") or config.get("token") or ""
        event = event_for(hook)
        if event is None or event["project"] in set(config.get("skip_projects") or []):
            return 0
        request = urllib.request.Request(
            url.rstrip("/") + "/announce", data=json.dumps(event).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "X-IFA-Token": token})
        urllib.request.urlopen(request, timeout=10).close()
    except Exception as exc:  # Ifa off, wrong code, no network: never bother Claude with it
        log = CONFIG.parent / "claude_hook.log"
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{type(exc).__name__}: {exc}\n")
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
