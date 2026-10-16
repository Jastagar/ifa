from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
SAFE = re.compile(r"^[A-Za-z0-9_./:@%+=,\\-]*$")


def _unquote(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1]
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return raw[1:-1].replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def quote(value: str) -> str:
    if SAFE.match(value):
        return value
    if "'" not in value:
        return f"'{value}'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = LINE.match(line)
        if match and match.group(1) not in values:
            values[match.group(1)] = _unquote(match.group(2))
    return values


def update_env(path: Path, changes: dict[str, str | None]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    written: set[str] = set()
    out: list[str] = []
    for line in lines:
        match = None if line.lstrip().startswith("#") else LINE.match(line)
        key = match.group(1) if match else None
        if key in changes:
            if key not in written and changes[key] is not None:
                out.append(f"{key}={quote(changes[key])}\n")
            written.add(key)
            continue
        out.append(line)

    added = [k for k, v in changes.items() if v is not None and k not in written]
    if added:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        if out and out[-1].strip():
            out.append("\n")
        out.extend(f"{k}={quote(changes[k])}\n" for k in added)

    _atomic_write(path, "".join(out))


def _atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd, tmp = tempfile.mkstemp(prefix=".env.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
