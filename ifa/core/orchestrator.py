import pathlib
import os
import queue
import sqlite3
import sys
import threading
import time
from ifa.core.agent_stream import (
    MODEL,
    agent_turn_stream,
)
from ifa.core.context import AgentContext
from ifa.core.dismiss import is_dismissal
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
from ifa.services.phone_server import PhoneTTS, start_phone_server
from ifa.tools import register_all
from ifa.tools.n8n import N8nConfigError, load_n8n_config
from ifa.voice.input import VoiceInput, init_input
from ifa.utils import latency
from ifa.utils.speech_queue import SpeechQueue
from concurrent.futures import ThreadPoolExecutor
from ifa.supervisor import RESTART_EXIT_CODE
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


def run_phone_turn(
    user_input: str,
    phone,
    ctx: AgentContext,
    memory: Memory,
    fillers: Fillers,
    executor: ThreadPoolExecutor,
) -> threading.Thread | None:
    phone_tts = PhoneTTS(ctx.tts, phone)

    def on_sentence(sentence: str) -> None:
        if not sentence.strip():
            return
        print(f"Ifa (phone): {sentence}")
        phone.send_reply(sentence)
        phone_tts.enqueue(sentence)

    pc_tts = ctx.tts
    ctx.tts = phone_tts
    try:
        timer = run_turn(user_input, ctx, memory, on_sentence, fillers, executor)
    except Exception as exc:
        print(f"[phone] turn failed: {exc}")
        phone.send_json({"type": "error", "message": "Ifa hit an error on that one."})
        phone.finish_turn(followup=False)
        return None
    finally:
        ctx.tts = pc_tts
    followup = not ctx.end_conversation

    def finish() -> None:
        phone_tts.wait_until_done()
        phone.finish_turn(followup=followup)
        timer.report()

    thread = threading.Thread(target=finish, name="ifa-phone-turn", daemon=True)
    thread.start()
    return thread


def _wait_until_quiet(speech_queue: SpeechQueue, tts: TTSService) -> None:
    speech_queue.wait_until_empty()
    while tts.is_speaking:
        time.sleep(0.05)


def _report_when_quiet(timer, speech_queue: SpeechQueue, tts: TTSService) -> None:
    _wait_until_quiet(speech_queue, tts)
    timer.report()


def _dismiss(ctx: AgentContext) -> None:
    ctx.turn += 1
    print("[ifa] conversation over (dismissed)")


def _restart() -> None:
    print("[ifa] restarting on request...", flush=True)
    sys.stdout.flush()
    os._exit(RESTART_EXIT_CODE)


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

    from ifa.voice import speaker
    print(f"[speaker] {speaker.shared().status()}")

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
    phones = None

    def announce_in_text_mode(level: str) -> None:
        print("\n🔔 New notification. Ask 'what is it?' to hear it.")
        if level == "urgent":
            announce_urgent()

    def route_announcement(level: str) -> None:
        if phones is not None and phones.nudge(level):
            return
        if isinstance(input_mode, VoiceInput):
            notifications.hold_announcement(level)
        else:
            announce_in_text_mode(level)

    notifications.on_announce = route_announcement
    notifications.start()

    VibeManager(tts)

    memory = Memory()

    def on_sentence(sentence: str):
        if not sentence.strip():
            return
        print(f"Ifa: {sentence}")
        speech_queue.enqueue(sentence)

    if os.environ.get("IFA_API_ENABLED", "1").lower() not in {"0", "false", "no", "off"}:
        start_activation_server(notifications)

    turns: queue.Queue = queue.Queue()

    def pump_local_input() -> None:
        while True:
            turns.put((input_mode.get(), None))

    threading.Thread(target=pump_local_input, name="ifa-local-input", daemon=True).start()
    phones = start_phone_server(
        lambda text, phone: turns.put((text, phone)), tts, notifications,
        urgent_clip=lambda: fillers.clip(URGENT_ANNOUNCEMENT),
    )
    if phones is not None:
        input_mode.pause_while(phones.has_priority)

    while True:
        raw_input, phone = turns.get()
        user_input = raw_input.strip()
        ctx.end_conversation = False
        ctx.restart_requested = False
        if phone is not None:
            print(f"Phone: {user_input}")
            if is_dismissal(user_input):
                _dismiss(ctx)
                phone.finish_turn(followup=False)
                continue
            finished = run_phone_turn(user_input, phone, ctx, memory, fillers, executor)
            if ctx.restart_requested:
                if finished is not None:
                    finished.join(timeout=30)
                _restart()
            continue
        if not user_input:
            continue
        print(f"You: {user_input}")
        if user_input.lower() in ["exit", "quit"]:
            break
        if is_dismissal(user_input):
            _dismiss(ctx)
            input_mode.end_turn()
            continue

        timer = run_turn(user_input, ctx, memory, on_sentence, fillers, executor)
        if ctx.restart_requested:
            _wait_until_quiet(speech_queue, tts)
            _restart()
        threading.Thread(
            target=_report_when_quiet, args=(timer, speech_queue, tts), daemon=True
        ).start()
        if ctx.end_conversation:
            input_mode.end_turn()
        else:
            input_mode.arm_followup()
