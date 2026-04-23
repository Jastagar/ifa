"""Main loop: read user input, run agent turn, speak response.

Stage 1 replaces the old `extract_fact → detect_intent → handle_with_intent
→ think → speak` chain with a single `agent_turn(user_text, ctx, memory)`
call. Tools (get_time, set_reminder, remember_fact, call_n8n_workflow) are
registered at startup and dispatched by the agent loop.

Startup order is load-bearing:
  1. Health-check Ollama (fail fast with actionable message)
  2. Load n8n config (graceful if missing; errors out on YAML syntax problems)
  3. init_db (WAL mode, schema retrofit)
  4. Construct TTSService + AgentContext
  5. register_all() tools
  6. resume_reminders() — reminder daemons can now fire via the same TTS
  7. Enter main loop
"""
import pathlib
import sqlite3
import sys
import threading
import time

from ifa.core.agent import MODEL, agent_turn
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.services.db import DB_PATH, init_db
from ifa.services.ollama_client import check_health
from ifa.services.tts_service import TTSService
from ifa.tools import register_all
from ifa.tools.n8n import N8nConfigError, load_n8n_config
from ifa.voice.input import get_input

N8N_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "n8n_workflows.yaml"


def resume_reminders(tts: TTSService, db_path: str) -> None:
    """Re-arm any reminders persisted in SQLite. Called once at startup."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    now = int(time.time())

    for reminder_id, task, trigger_time in c.execute(
        "SELECT id, task, trigger_time FROM reminders"
    ):
        delay = max(0, trigger_time - now)

        def worker(reminder_id=reminder_id, task=task, delay=delay, db_path=db_path):
            time.sleep(delay)
            message = f"Reminder: {task}"
            print(f"\n⏰ {message}")
            tts.speak(message)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            conn.commit()
            conn.close()

        threading.Thread(target=worker, daemon=True).start()

    conn.close()


def run() -> None:
    print("Orchestrator running...")

    # 1. Ollama health check (fail fast)
    try:
        check_health(required_model=MODEL)
    except RuntimeError as exc:
        print(f"\n❌ {exc}\n", file=sys.stderr)
        sys.exit(1)

    # 2. n8n config load (graceful if missing; hard error on syntax issues)
    try:
        n8n_config = load_n8n_config(N8N_CONFIG_PATH)
    except N8nConfigError as exc:
        print(f"\n❌ {exc}\n", file=sys.stderr)
        sys.exit(1)

    if n8n_config:
        print(f"Loaded {len(n8n_config)} n8n workflow(s): {sorted(n8n_config.keys())}")
    else:
        print(f"No n8n workflows configured (expected at {N8N_CONFIG_PATH}).")

    # 3-5. DB init, TTS, register tools
    init_db()
    tts = TTSService()
    ctx = AgentContext(tts=tts, db_path=DB_PATH, n8n_config=n8n_config)
    register_all()

    # 6. Restore reminders BEFORE entering the main loop
    resume_reminders(tts, DB_PATH)

    # 7. Main loop
    memory = Memory()
    while True:
        user_input = get_input().strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        reply = agent_turn(user_input, ctx, memory)
        print("Ifa:", reply)
        tts.speak(reply)
