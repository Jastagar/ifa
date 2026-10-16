import sqlite3

DB_PATH = "ifa.db"

def init_db(db_path: str | None = None) -> None:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    c = conn.cursor()

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