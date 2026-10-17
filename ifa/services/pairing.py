from __future__ import annotations

import os
import pathlib
import secrets
import socket

TOKEN_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / ".phone_token"


def load_pairing_code(path: pathlib.Path = TOKEN_PATH, configured: str | None = None) -> str:
    code = (configured if configured is not None else os.environ.get("IFA_PHONE_TOKEN", "")).strip()
    if code:
        return code
    if path.exists():
        code = path.read_text(encoding="utf-8").strip()
        if code:
            return code
    code = f"{secrets.randbelow(10 ** 6):06d}"
    path.write_text(code, encoding="utf-8")
    return code


def lan_address() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            return sock.getsockname()[0]
    except OSError:
        return None
