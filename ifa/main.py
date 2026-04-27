"""Ifa entry point.

Loads ``.env`` (if present) at the top BEFORE any module reads env
vars, then hands off to the orchestrator. Settings already exported
in the parent shell / launcher take precedence over .env (the
default ``override=False`` semantics of python-dotenv).
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
