"""Ifa entry point.

Loads ``.env`` (if present) at the top BEFORE any module reads env
vars, then hands off to the orchestrator. Settings already exported
in the parent shell / launcher take precedence over .env (the
default ``override=False`` semantics of python-dotenv).
"""
import pathlib

from dotenv import load_dotenv

# Load .env from the repo root (one level up from this package).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)

from ifa.core.orchestrator import run  # noqa: E402 — import after env load

if __name__ == "__main__":
    print("Ifa booting up...")
    run()
