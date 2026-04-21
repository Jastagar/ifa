from ifa.voice.input import get_input
from ifa.core.brain import think, detect_intent, extract_fact
from ifa.skills.manager import handle_with_intent
from ifa.services.tts_service import TTSService
from ifa.services.db import init_db, DB_PATH
import threading
import sqlite3
import time


tts = TTSService()


def resume_reminders():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = int(time.time())

    for reminder_id, task, trigger_time in c.execute(
        "SELECT id, task, trigger_time FROM reminders"
    ):
        delay = max(0, trigger_time - now)

        def worker(reminder_id=reminder_id, task=task, delay=delay):
            time.sleep(delay)

            message = f"Reminder: {task}"
            print(f"\n⏰ {message}")
            tts.speak(message)

            # ✅ delete after firing
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            conn.commit()
            conn.close()

        threading.Thread(target=worker, daemon=True).start()

    conn.close()


def run():
    print("Orchestrator running...")

    # ✅ setup DB + restore reminders
    init_db()
    resume_reminders()

    while True:
        user_input = get_input().strip()

        fact = extract_fact(user_input)

        if fact:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO facts (fact) VALUES (?)", (fact,))
            conn.commit()
            conn.close()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        intent = detect_intent(user_input)

        skill_response = handle_with_intent(intent, user_input)

        if skill_response:
            response = skill_response
        else:
            response = think(user_input)

        print("Ifa:", response)
        tts.speak(response)