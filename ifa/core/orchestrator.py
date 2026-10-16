import pathlib
import os
import sqlite3
import sys
import threading
import time
from ifa.core.agent_stream import (
    MODEL,
    agent_turn_stream,
)
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.services.db import DB_PATH, init_db
from ifa.services.fillers import Fillers, generic_phrases
from ifa.services.notifications import (
    URGENT_ANNOUNCEMENT,
    NotificationCenter,
    NotificationInbox,
    classify_urgency,
    summarize_bursts,
)
from ifa.services.ollama_client import chat, check_health
from ifa.services.tts_service import TTSService
from ifa.services.activation_server import start_activation_server
from ifa.tools import register_all
from ifa.tools.n8n import N8nConfigError, load_n8n_config
from ifa.voice.input import VoiceInput, init_input
from ifa.utils import latency
from ifa.utils.speech_queue import SpeechQueue
from concurrent.futures import ThreadPoolExecutor
from ifa.skills.vibe.vibe import VibeManager


N8N_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "n8n_workflows.yaml"

CONTACTS = {
    "john":"+918876700414",
}


FILLER_DELAY_SEC = int(os.environ.get("IFA_FILLER_DELAY_MS", "650")) / 1000


def run_turn(
    user_input: str,
    ctx: AgentContext,
    memory: Memory,
    on_sentence,
    fillers: Fillers,
    executor: ThreadPoolExecutor,
    filler_delay: float = FILLER_DELAY_SEC,
) -> latency.TurnTimer:
    timer = latency.claim_or_begin("text")
    answer_started = threading.Event()
    filler_played = False
    speech_order = threading.Lock()

    def play_filler(tool_names: list[str] | None = None) -> None:
        nonlocal filler_played
        with speech_order:
            if answer_started.is_set() or filler_played:
                return
            choice = fillers.choose(tool_names)
            if choice is None:
                return
            phrase, audio = choice
            filler_played = True
            timer.filler_text = phrase
            timer.mark("filler_queued")
            print(f"Ifa: ({phrase})")
            ctx.tts.play_clip(audio, phrase)

    def on_answer_sentence(sentence: str) -> None:
        with speech_order:
            timer.mark("first_sentence")
            answer_started.set()
            on_sentence(sentence)

    agent_future = executor.submit(
        agent_turn_stream, user_input, ctx, memory, on_answer_sentence, play_filler
    )
    if not answer_started.wait(timeout=filler_delay):
        play_filler()
    agent_future.result()
    return timer


def _report_when_quiet(timer, speech_queue: SpeechQueue, tts: TTSService) -> None:
    speech_queue.wait_until_empty()
    while tts.is_speaking:
        time.sleep(0.05)
    timer.report()


def resume_reminders(tts: TTSService, db_path: str) -> None:
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

    if not MODEL:
        print(
            "\n❌ IFA_OLLAMA_MODEL is not set. Add it to .env, for example: "
            "IFA_OLLAMA_MODEL=your-model-name\n",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        check_health(required_model=MODEL)
    except RuntimeError as exc:
        print(f"\n❌ {exc}\n", file=sys.stderr)
        sys.exit(1)

    try:
        n8n_config = load_n8n_config(N8N_CONFIG_PATH)
    except N8nConfigError as exc:
        print(f"\n❌ {exc}\n", file=sys.stderr)
        sys.exit(1)

    if n8n_config:
        print(f"Loaded {len(n8n_config)} n8n workflow(s): {sorted(n8n_config.keys())}")
    else:
        print(f"No n8n workflows configured (expected at {N8N_CONFIG_PATH}).")

    init_db()
    tts = TTSService()
    speech_queue = SpeechQueue(
        handler=tts.enqueue
    )
    executor = ThreadPoolExecutor(max_workers=1)
    fillers = Fillers(tts.synthesize, generic=generic_phrases(), extra=[URGENT_ANNOUNCEMENT])
    fillers.prepare_in_background()
    notifications = NotificationCenter(
        NotificationInbox(),
        classify=lambda burst: classify_urgency(burst, chat, MODEL),
        summarize=lambda bursts: summarize_bursts(bursts, chat, MODEL),
    )
    ctx = AgentContext(
        tts=tts, db_path=DB_PATH, n8n_config=n8n_config, contacts=CONTACTS,
        notifications=notifications,
    )
    register_all()

    resume_reminders(tts, DB_PATH)

    def announce_urgent() -> None:
        audio = fillers.clip(URGENT_ANNOUNCEMENT)
        if audio is not None:
            done = tts.play_clip(audio, URGENT_ANNOUNCEMENT)
        else:
            done = tts.enqueue(URGENT_ANNOUNCEMENT)
        if done:
            done.wait()

    input_mode = init_input(tts, notifications, announce_urgent)
    if not isinstance(input_mode, VoiceInput):
        def announce_in_text_mode(level: str) -> None:
            print("\n🔔 New notification. Ask 'what is it?' to hear it.")
            if level == "urgent":
                announce_urgent()

        notifications.on_announce = announce_in_text_mode
    notifications.start()

    VibeManager(tts)

    memory = Memory()

    def on_sentence(sentence: str):
        print(f"Ifa: {sentence}")
        speech_queue.enqueue(sentence)

    if os.environ.get("IFA_API_ENABLED", "1").lower() not in {"0", "false", "no", "off"}:
        start_activation_server(notifications)

    while True:
        user_input = input_mode.get().strip()
        if not user_input:
            continue
        print(f"You: {user_input}")
        if user_input.lower() in ["exit", "quit"]:
            break

        timer = run_turn(user_input, ctx, memory, on_sentence, fillers, executor)
        threading.Thread(
            target=_report_when_quiet, args=(timer, speech_queue, tts), daemon=True
        ).start()
        input_mode.arm_followup()
