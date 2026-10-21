from __future__ import annotations

import os
from contextlib import closing
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS notification_history (
    id INTEGER PRIMARY KEY,
    received_at REAL NOT NULL,
    app TEXT NOT NULL,
    package TEXT NOT NULL,
    sender TEXT NOT NULL,
    text TEXT NOT NULL,
    priority TEXT NOT NULL
)
"""


class NotificationHistory:
    """Every forwarded notification, with the app that sent it, kept in ifa.db for IFA_NOTIF_HISTORY_DAYS."""

    def __init__(self, db_path: str, keep_days: float | None = None, clock=time.time) -> None:
        self.db_path = db_path
        self.keep_days = keep_days if keep_days is not None else float(os.environ.get("IFA_NOTIF_HISTORY_DAYS", "30"))
        self._clock = clock
        self._lock = threading.Lock()
        with closing(self._connect()) as conn, conn:
            conn.execute(SCHEMA)
            conn.execute("CREATE INDEX IF NOT EXISTS notification_history_at ON notification_history (received_at)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5)

    def record(self, app: str, package: str, sender: str, text: str, priority: str) -> None:
        if self.keep_days <= 0:
            return
        now = self._clock()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO notification_history (received_at, app, package, sender, text, priority) VALUES (?, ?, ?, ?, ?, ?)",
                (now, app, package, sender, text, priority),
            )
            conn.execute("DELETE FROM notification_history WHERE received_at < ?", (now - self.keep_days * 86400,))

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT received_at, app, package, sender, text, priority FROM notification_history "
                "ORDER BY received_at DESC, id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
