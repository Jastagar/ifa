import threading
import time
import sqlite3

from ifa.skills.base import Skill
from ifa.services.db import DB_PATH


class ReminderSkill(Skill):
    def __init__(self, tts, db_path: str | None = None):
        self.tts = tts
        self.db_path = db_path or DB_PATH

    def can_handle(self, text: str) -> bool:
        t = text.lower()
        return "remind" in t or "tell me" in t

    def schedule(self, task: str, seconds: int) -> str:
        """Store a reminder in SQLite and spawn the firing daemon thread.

        This is the tool-callable entry point. JSON-Schema validation at the
        registry layer already confirmed `task: str` and `seconds: int`;
        this method only guards the business rule (positive delay).
        """
        if seconds <= 0:
            return "Time must be greater than zero."

        trigger_time = int(time.time()) + seconds

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO reminders (task, trigger_time) VALUES (?, ?)",
            (task, trigger_time),
        )
        reminder_id = c.lastrowid
        conn.commit()
        conn.close()

        threading.Thread(
            target=self._reminder,
            args=(task, seconds, reminder_id),
            daemon=True,
        ).start()

        return f"Okay, I will remind you to {task} in {seconds} seconds."

    def _reminder(self, task, seconds, reminder_id=None):
        time.sleep(seconds)

        message = f"Reminder: {task}"
        print(f"\n⏰ {message}")
        self.tts.speak(message)

        # ✅ delete after firing
        if reminder_id:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            conn.commit()
            conn.close()