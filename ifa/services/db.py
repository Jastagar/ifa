import sqlite3
from pathlib import Path

from ifa.services import memory_store

DB_PATH = "ifa.db"
MEMORY_MARKDOWN = Path("memory.md")

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

    c.execute(memory_store.SCHEMA)

    conn.commit()
    conn.close()

    imported = memory_store.import_markdown(path, MEMORY_MARKDOWN)
    if imported:
        print(f"[memory] imported {imported} memories from {MEMORY_MARKDOWN.name} (renamed to {MEMORY_MARKDOWN.name}.imported)")