"""Ifa entry point — first thing run by ``python -m ifa.main``.

If you're learning the codebase, read this file first, then
``ifa/core/orchestrator.py`` (the main loop), then
``ifa/core/agent.py`` (the LLM tool-call loop). After that, peek into
``ifa/voice/wake_word.py`` and ``ifa/voice/input.py`` for the voice
pipeline. ``docs/ARCHITECTURE.md`` has the full call graph if you want
the bird's-eye view first.

What this file does:

1. **Load ``.env``** at the very top, before anything else imports.
   Order matters — every module reads env vars at import time
   (``os.environ.get(...)`` calls), so the .env values must be in
   ``os.environ`` before those reads happen. ``override=False`` means
   shell-exported / launcher-set values still win, which is the
   semantics we want — a developer running ``IFA_WHISPER_DEVICE=cpu
   python -m ifa.main`` should override the .env default.

2. **Print an env summary** so the developer can see at a glance which
   ``IFA_*`` knobs are actually in effect. Useful for debugging "why
   isn't my .env change taking effect?" — if the variable shows up in
   this dump with the right value, the runtime saw it.

3. **Hand off to the orchestrator** (``ifa.core.orchestrator.run``).
   That function does the actual startup: Ollama health check, n8n
   config load, DB init, TTS setup, tool registration, reminder
   recovery, then the main loop.

Why these three steps live in main.py and not orchestrator.run:

- ``.env`` loading must happen before ANY ``ifa.*`` import — including
  the orchestrator's own imports. So it can't live in run().
- The env summary is debugging output for the developer, not part of
  the assistant's behavior. Keeping it at the entry point makes the
  separation clear.
- The orchestrator stays focused on "starting Ifa", not on "how was
  this process launched".
"""
import os
import pathlib

from dotenv import load_dotenv

# Load .env from the repo root (one level up from this package).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DOTENV_PATH = _REPO_ROOT / ".env"
_dotenv_loaded = load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


def _log_env_summary() -> None:
    """Print every IFA_* (and a few HF_*) env var in effect, plus where
    .env was loaded from. Run early so the user can confirm at a glance
    which configuration is actually in use this session.

    Tokens are masked.
    """
    if _dotenv_loaded:
        print(f"[env] loaded {_DOTENV_PATH}")
    else:
        print(f"[env] no .env file at {_DOTENV_PATH}  (using shell + defaults)")

    interesting_prefixes = ("IFA_",)
    interesting_exact = {"HF_HUB_OFFLINE", "HF_TOKEN", "PYTHONPATH"}

    rows: list[tuple[str, str]] = []
    for k, v in sorted(os.environ.items()):
        if k.startswith(interesting_prefixes) or k in interesting_exact:
            display = v
            # Mask any value that looks like a credential
            if k == "HF_TOKEN" and v:
                display = f"{v[:4]}…{v[-4:]} (masked)" if len(v) > 8 else "(set, masked)"
            rows.append((k, display))

    if not rows:
        print("[env] no IFA_* / HF_* variables set; using all defaults")
        return

    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"[env] {k.ljust(width)} = {v}")


_log_env_summary()


from ifa.core.orchestrator import run  # noqa: E402 — import after env load + log

if __name__ == "__main__":
    print("Ifa booting up...")
    run()
