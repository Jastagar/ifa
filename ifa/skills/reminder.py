import threading
import time
import sqlite3

from ifa.services.db import DB_PATH


class ReminderSkill:
    def __init__(self, tts, db_path: str | None = None):
        self.tts = tts
        self.db_path = db_path or DB_PATH

    def schedule(self, task: str, seconds: int) -> str:
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

        if reminder_id:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            conn.commit()
            conn.close()