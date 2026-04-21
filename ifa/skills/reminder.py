import threading
import time
import sqlite3

from ifa.skills.base import Skill
from ifa.core.brain import extract_reminder
from ifa.services.db import DB_PATH


class ReminderSkill(Skill):
    def __init__(self, tts):
        self.tts = tts

    def can_handle(self, text: str) -> bool:
        t = text.lower()
        return "remind" in t or "tell me" in t

    def handle(self, text: str) -> str:
        data = extract_reminder(text)

        task = data.get("task")
        seconds = data.get("seconds")

        # ✅ validate
        if not task or not isinstance(task, str):
            return "I couldn't understand the reminder."

        try:
            seconds = int(seconds)
        except:
            return "I couldn't understand the time for the reminder."

        if seconds <= 0:
            return "Time must be greater than zero."

        # ✅ store in DB FIRST
        trigger_time = int(time.time()) + seconds

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute(
            "INSERT INTO reminders (task, trigger_time) VALUES (?, ?)",
            (task, trigger_time)
        )

        reminder_id = c.lastrowid

        conn.commit()
        conn.close()

        # ✅ THEN start thread
        threading.Thread(
            target=self._reminder,
            args=(task, seconds, reminder_id),
            daemon=True
        ).start()

        return f"Okay, I will remind you to {task} in {seconds} seconds."

    def _reminder(self, task, seconds, reminder_id=None):
        time.sleep(seconds)

        message = f"Reminder: {task}"
        print(f"\n⏰ {message}")
        self.tts.speak(message)

        # ✅ delete after firing
        if reminder_id:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            conn.commit()
            conn.close()