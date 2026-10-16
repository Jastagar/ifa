from __future__ import annotations

import os


OLLAMA_MODEL = os.environ.get("IFA_OLLAMA_MODEL", "").strip() or None

MAIN_OLLAMA_URL =  os.environ.get("MAIN_OLLAMA_URL", "").strip() or "http://localhost:11434"

OLLAMA_THINK = os.environ.get("IFA_OLLAMA_THINK", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

OLLAMA_KEEP_ALIVE = os.environ.get("IFA_OLLAMA_KEEP_ALIVE", "10m").strip()
