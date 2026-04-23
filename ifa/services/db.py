import sqlite3

DB_PATH = "ifa.db"

def init_db(db_path: str | None = None) -> None:
    """Initialize schema. Safe to call repeatedly.

    Applies WAL journal mode to reduce write-write contention between the
    reminder daemon thread and main-thread tool handlers. Retrofits the
    UNIQUE index on `facts.fact` if the DB pre-dates the constraint (older
    Ifa installs may have a facts table without UNIQUE).
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

    c.execute("""
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY,
        fact TEXT UNIQUE
    )
    """)

    # Retrofit UNIQUE index for pre-existing facts tables that lacked it.
    # CREATE TABLE IF NOT EXISTS doesn't alter an existing schema.
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_unique ON facts(fact)")

    conn.commit()
    conn.close()