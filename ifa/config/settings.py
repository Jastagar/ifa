"""Runtime settings sourced from environment variables."""
from __future__ import annotations

import os


# The selected Ollama model lives exclusively in IFA_OLLAMA_MODEL (normally
# set in .env). Keeping this value optional lets unit tests import modules
# without requiring a user-specific .env file; startup validates it.
OLLAMA_MODEL = os.environ.get("IFA_OLLAMA_MODEL", "").strip() or None

# Qwen3 enables an internal reasoning pass by default. That is useful for
# difficult problems, but it makes short voice-assistant turns noticeably
# slower. Set IFA_OLLAMA_THINK=1 to trade latency for more deliberate answers.
OLLAMA_THINK = os.environ.get("IFA_OLLAMA_THINK", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# Keep the model resident between turns. Reloading a 5.6 GB model after a
# short idle period is much more noticeable than the small amount of VRAM it
# consumes while Ifa is running.
OLLAMA_KEEP_ALIVE = os.environ.get("IFA_OLLAMA_KEEP_ALIVE", "10m").strip()
