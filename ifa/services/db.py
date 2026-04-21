import sqlite3

DB_PATH = "ifa.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # reminders
    c.execute("""
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY,
        task TEXT,
        trigger_time INTEGER
    )
    """)

    # facts
    c.execute("""
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY,
        fact TEXT UNIQUE
    )
    """)

    conn.commit()
    conn.close()