import os
import pathlib
import sys

from ifa.supervisor import is_supervised, supervise

if __name__ == "__main__" and not is_supervised():
    sys.exit(supervise())


from dotenv import load_dotenv

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DOTENV_PATH = _REPO_ROOT / ".env"
_dotenv_loaded = load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


def _log_env_summary() -> None:
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


from ifa.core.orchestrator import run

if __name__ == "__main__":
    print("Ifa booting up...")
    run()
