"""Reminder skill — persists a task in SQLite, fires a TTS notification later.

The only method called in production is ``schedule(task, seconds)``, by
the ``set_reminder`` tool adapter in ``ifa/tools/reminder.py``. The
``Skill``-base ``can_handle``/``handle`` interface is vestigial (see
``ifa/skills/base.py`` for why).

Lifecycle of a reminder
-----------------------
1. LLM emits a ``set_reminder`` tool call. The registry validates args
   against the JSON Schema, then calls the adapter, which calls
   ``ReminderSkill.schedule(task, seconds)``.
2. ``schedule`` inserts a row into ``reminders`` (SQLite, via WAL mode
   so the daemon thread can read while the main thread writes), then
   spawns a daemon thread that sleeps ``seconds`` and fires the
   notification.
3. When the timer expires, ``_reminder`` prints + TTS-speaks the
   reminder, then deletes the row so a subsequent
   ``resume_reminders`` boot doesn't re-arm it.
4. If Ifa is killed before the timer fires, the row survives. On next
   startup, ``orchestrator.resume_reminders`` reads pending rows and
   re-arms a daemon for each — so a reminder set "in 24 hours" still
   fires even if you reboot the box mid-wait.

Why daemon threads, not asyncio?
--------------------------------
Stage 1 is fully synchronous. A daemon thread per reminder is dead
simple, leaks no resources at process exit, and avoids dragging an
event loop into the orchestrator. With dozens of pending reminders
this would be wasteful, but for personal use (a handful at most),
it's the YAGNI choice.
"""
import threading
import time
import sqlite3

from ifa.skills.base import Skill
from ifa.services.db import DB_PATH


class ReminderSkill(Skill):
    """Schedules and fires reminders. Only ``schedule()`` is called externally."""

    def __init__(self, tts, db_path: str | None = None):
        self.tts = tts
        self.db_path = db_path or DB_PATH

    def can_handle(self, text: str) -> bool:
        # Vestigial — never called in Stage 1. See module docstring.
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