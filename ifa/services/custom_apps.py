from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APPS_FILE = REPO_ROOT / "apps.json"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,39}$")
MAX_COMMAND = 1000
MAX_APPS = 100


def validate(raw) -> tuple[list[dict], dict[int, str]]:
    if not isinstance(raw, list):
        raise ValueError("apps must be a list")
    if len(raw) > MAX_APPS:
        raise ValueError(f"at most {MAX_APPS} apps")
    apps, errors, seen = [], {}, set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors[i] = "must be an object with name and command"
            continue
        name = " ".join(str(item.get("name") or "").split())
        command = str(item.get("command") or "").strip()
        if not NAME.match(name):
            errors[i] = "name: 1-40 letters, digits, spaces or . _ + -"
        elif name.lower() in seen:
            errors[i] = f"{name} is listed twice"
        elif not command:
            errors[i] = "command is empty"
        elif len(command) > MAX_COMMAND or "\n" in command or "\r" in command:
            errors[i] = f"command must be one line of at most {MAX_COMMAND} characters"
        else:
            seen.add(name.lower())
            apps.append({"name": name, "command": command})
    return apps, errors


def load(path: Path = APPS_FILE) -> list[dict]:
    if not path.exists():
        return []
    try:
        apps, errors = validate(json.loads(path.read_text(encoding="utf-8")).get("apps", []))
    except (OSError, ValueError, AttributeError) as exc:
        print(f"[apps] ignoring {path.name}: {exc}")
        return []
    for i, message in errors.items():
        print(f"[apps] skipping entry {i + 1} in {path.name}: {message}")
    return apps


def save(apps: list[dict], path: Path = APPS_FILE) -> None:
    data = json.dumps({"apps": apps}, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".apps-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
