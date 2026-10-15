import sqlite3

DB_PATH = "ifa.db"

def init_db(db_path: str | None = None) -> None:
    """Initialize schema. Safe to call repeatedly.

    Applies WAL journal mode to reduce write-write contention between the
    reminder daemon thread and main-thread tool handlers.
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    c = conn.cursor()

    # Reduce contention on concurrent writers. WAL survives across connections.
    c.execute("PRAGMA journal_mode=WAL")

    c.execute("""
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY,
        task TEXT,
        trigger_time INTEGER
    )
    """)

    conn.commit()
    conn.close()