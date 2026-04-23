import sqlite3
import threading
import time

from ifa.core.brain import detect_intent, extract_fact, think
from ifa.services.db import DB_PATH, init_db
from ifa.services.tts_service import TTSService
from ifa.skills.manager import handle_with_intent
from ifa.voice.input import get_input


def resume_reminders(tts):
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

            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            conn.commit()
            conn.close()

        threading.Thread(target=worker, daemon=True).start()

    conn.close()


def run():
    print("Orchestrator running...")

    tts = TTSService()
    init_db()
    resume_reminders(tts)

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

        skill_response = handle_with_intent(intent, user_input, tts)

        if skill_response:
            response = skill_response
        else:
            response = think(user_input)

        print("Ifa:", response)
        tts.speak(response)
