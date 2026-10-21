from __future__ import annotations

import os
from pathlib import Path

WHISPER_DEFAULT = "Oriserve/Whisper-Hindi2Hinglish-Apex"
CHATTERBOX_REPO = "ResembleAI/chatterbox-turbo"
OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return home / "hub"


def required_repos() -> list[str]:
    whisper = os.environ.get("IFA_WHISPER_MODEL", "").strip() or WHISPER_DEFAULT
    return [whisper, CHATTERBOX_REPO]


def is_cached(repo_id: str, cache: Path | None = None) -> bool:
    if os.path.isdir(repo_id):
        return True
    snapshots = (cache or hf_cache_dir()) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    for snapshot in snapshots.iterdir():
        names = [p.name for p in snapshot.rglob("*") if p.is_file()]
        has_config = any(n.endswith(".json") for n in names)
        has_weights = any(n.endswith((".safetensors", ".bin", ".pt")) for n in names)
        if has_config and has_weights:
            return True
    return False


def configure(cache: Path | None = None) -> str:
    mode = os.environ.get("IFA_OFFLINE", "auto").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        offline, why = True, "IFA_OFFLINE=on"
    elif mode in {"0", "false", "no", "off"}:
        offline, why = False, "IFA_OFFLINE=off"
    else:
        missing = [repo for repo in required_repos() if not is_cached(repo, cache)]
        offline = not missing
        why = ("all models are downloaded" if offline else
               f"not downloaded yet: {', '.join(missing)}. They download on first use, which needs the internet "
               "(launcher: Setup, Download voice models)")
    for var in OFFLINE_VARS:
        os.environ[var] = "1" if offline else "0"
    return f"{'offline: models load from disk, no internet needed' if offline else 'online'} ({why})"
