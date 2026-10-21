from __future__ import annotations

import json
import platform
import re
import ssl
import time
import urllib.error
import urllib.request

PASTE_URL = "https://paste.rs/"
MAX_CHARS = 900_000
TOKEN_PATTERNS = [
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}"),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bya29\.[A-Za-z0-9_.-]{20,}"),
    re.compile(r"(?i)(pairing code[:\s]+)\d{4,8}"),
    re.compile(r"(?i)(X-IFA-Token[\"':\s]+)[^\s\"']+"),
]


def redact(text: str, secrets: list[str]) -> str:
    for value in sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True):
        text = text.replace(value, "[secret]")
    for pattern in TOKEN_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[secret]", text)
    return text


def build_report(log_text: str, checks: dict, setup: dict, secrets: list[str]) -> str:
    steps = "\n".join(
        f"  {'done   ' if s.get('done') else 'blocked' if s.get('blocked') else 'todo   '} {s['label']}"
        + (f"  ({s['blocked']})" if s.get("blocked") else "")
        for s in setup.get("steps", [])
    )
    header = (
        f"Ifa logs, {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Machine: {platform.platform()} | launcher Python {platform.python_version()}\n"
        f"Checks: {json.dumps(checks, default=str)}\n"
        f"Setup steps:\n{steps}\n"
        f"{'=' * 70}\n"
    )
    body = log_text[-MAX_CHARS:]
    return redact(header + body + "\n", secrets)


def upload(text: str, url: str = PASTE_URL, timeout: float = 30) -> str:
    request = urllib.request.Request(url, data=text.encode("utf-8"), method="POST",
                                     headers={"Content-Type": "text/plain; charset=utf-8", "User-Agent": "ifa-launcher"})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        if not isinstance(getattr(exc, "reason", None), ssl.SSLError):
            raise
        import certifi
        response = urllib.request.urlopen(request, timeout=timeout,
                                          context=ssl.create_default_context(cafile=certifi.where()))
    with response:
        link = response.read().decode().strip()
    if not link.startswith("http"):
        raise OSError(f"unexpected answer from the paste service: {link[:100]}")
    return link
