from __future__ import annotations

import re
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path

KINDS = ("progress", "preference", "instruction", "fact")
ALWAYS_KNOWN = ("preference", "instruction")
MAX_TOPIC = 60
MAX_FACT = 500
MAX_ALWAYS_KNOWN = 25
MAX_TOPICS_LISTED = 40
SIMILAR_TOPIC = 0.82
STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "mine", "was", "were", "is", "are", "am", "on", "in", "of", "to", "for",
    "with", "at", "about", "what", "where", "which", "who", "how", "when", "do", "did", "does", "you", "your",
    "remember", "tell", "know", "sir", "ifa", "please", "again", "last", "time", "left", "it", "that", "this",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    topic TEXT NOT NULL,
    key TEXT NOT NULL UNIQUE,
    fact TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    previous TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def topic_key(topic: str) -> str:
    words = re.sub(r"[^a-z0-9 ]+", " ", topic.lower()).split()
    return " ".join(w for w in words if w not in {"the", "a", "an", "my"}) or " ".join(words)


def _words(text: str) -> set[str]:
    return {w for w in re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split() if w not in STOPWORDS}


def _clean(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]


def _similar_row(conn: sqlite3.Connection, key: str) -> sqlite3.Row | None:
    row = conn.execute("SELECT * FROM memories WHERE key = ?", (key,)).fetchone()
    if row is not None:
        return row
    best, best_score = None, 0.0
    for candidate in conn.execute("SELECT * FROM memories"):
        other = candidate["key"]
        contained = (f" {key} " in f" {other} " or f" {other} " in f" {key} ") and min(len(key), len(other)) >= 4
        score = 1.0 if contained else SequenceMatcher(None, key, other).ratio()
        if score > best_score:
            best, best_score = candidate, score
    return best if best_score >= SIMILAR_TOPIC else None


def remember(db_path: str, topic: str, fact: str, kind: str = "fact", now: float | None = None) -> dict:
    topic, fact = _clean(topic, MAX_TOPIC), _clean(fact, MAX_FACT)
    if not topic or not fact:
        raise ValueError("a memory needs a topic and a fact")
    kind = kind if kind in KINDS else "fact"
    now = int(now if now is not None else time.time())
    with _connect(db_path) as conn:
        existing = _similar_row(conn, topic_key(topic))
        if existing is None:
            conn.execute(
                "INSERT INTO memories (topic, key, fact, kind, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (topic, topic_key(topic), fact, kind, now, now),
            )
            return {"topic": topic, "fact": fact, "kind": kind, "previous": None}
        previous = existing["fact"] if existing["fact"] != fact else existing["previous"]
        conn.execute(
            "UPDATE memories SET fact = ?, kind = ?, previous = ?, updated_at = ? WHERE id = ?",
            (fact, kind, previous, now, existing["id"]),
        )
        return {"topic": existing["topic"], "fact": fact, "kind": kind,
                "previous": existing["fact"] if existing["fact"] != fact else None}


def recall(db_path: str, query: str, limit: int = 3) -> list[dict]:
    wanted = _words(query)
    key = topic_key(query)
    with _connect(db_path) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM memories")]
    scored = []
    for row in rows:
        topic_words, fact_words = _words(row["topic"]), _words(row["fact"])
        score = 2 * len(wanted & topic_words) + len(wanted & fact_words)
        score += 2 * SequenceMatcher(None, key, row["key"]).ratio() if key else 0
        if wanted & (topic_words | fact_words) or SequenceMatcher(None, key, row["key"]).ratio() >= 0.6:
            scored.append((score, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:limit]]


def forget(db_path: str, topic: str) -> dict | None:
    with _connect(db_path) as conn:
        row = _similar_row(conn, topic_key(topic))
        if row is None:
            return None
        conn.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
        return dict(row)


def all_memories(db_path: str) -> list[dict]:
    with _connect(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM memories ORDER BY updated_at DESC")]


def validate(raw) -> tuple[list[dict], dict[int, str]]:
    if not isinstance(raw, list):
        raise ValueError("memories must be a list")
    items, errors, seen = [], {}, set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors[i] = "must be an object"
            continue
        topic, fact = _clean(item.get("topic") or "", MAX_TOPIC), _clean(item.get("fact") or "", MAX_FACT)
        kind = item.get("kind") or "fact"
        if not topic or not fact:
            errors[i] = "needs a topic and what to remember"
        elif kind not in KINDS:
            errors[i] = f"kind must be one of {', '.join(KINDS)}"
        elif topic_key(topic) in seen:
            errors[i] = f"{topic} is listed twice"
        else:
            seen.add(topic_key(topic))
            items.append({"id": item.get("id"), "topic": topic, "fact": fact, "kind": kind})
    return items, errors


def replace_all(db_path: str, items: list[dict], now: float | None = None) -> None:
    now = int(now if now is not None else time.time())
    with _connect(db_path) as conn:
        existing = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM memories")}
        keep = {item["id"] for item in items if item.get("id") in existing}
        for old_id in set(existing) - keep:
            conn.execute("DELETE FROM memories WHERE id = ?", (old_id,))
        conn.execute("UPDATE memories SET key = '~' || id")
        for item in items:
            old = existing.get(item.get("id"))
            if old is None:
                conn.execute(
                    "INSERT INTO memories (topic, key, fact, kind, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (item["topic"], topic_key(item["topic"]), item["fact"], item["kind"], now, now),
                )
                continue
            changed = (old["topic"], old["fact"], old["kind"]) != (item["topic"], item["fact"], item["kind"])
            conn.execute(
                "UPDATE memories SET topic = ?, key = ?, fact = ?, kind = ?, previous = ?, updated_at = ? WHERE id = ?",
                (item["topic"], topic_key(item["topic"]), item["fact"], item["kind"],
                 old["fact"] if old["fact"] != item["fact"] else old["previous"],
                 now if changed else old["updated_at"], old["id"]),
            )


def ago(timestamp: int, now: float | None = None) -> str:
    days = int(((now if now is not None else time.time()) - timestamp) // 86400)
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return time.strftime("on %d %B %Y", time.localtime(timestamp))


def prompt_context(db_path: str) -> str:
    try:
        rows = all_memories(db_path)
    except sqlite3.Error as exc:
        print(f"[memory] can't read memories: {exc}")
        return ""
    known = [r for r in rows if r["kind"] in ALWAYS_KNOWN][:MAX_ALWAYS_KNOWN]
    others = [r["topic"] for r in rows if r["kind"] not in ALWAYS_KNOWN][:MAX_TOPICS_LISTED]
    parts = []
    if known:
        lines = "\n".join(f"- {r['topic']}: {r['fact']}" for r in known)
        parts.append(f"Sir's preferences and standing instructions (follow them):\n{lines}")
    if others:
        parts.append(
            "You also have saved notes on: " + ", ".join(others) + ". "
            "Call recall before answering anything about them; never guess."
        )
    return "\n\n".join(parts)


def import_markdown(db_path: str, path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^- (?:\[[^\]]*\]\s*)?(.+)$", line.strip())
        if not match:
            continue
        fact = match.group(1).strip()
        topic = " ".join(fact.split()[:5])
        try:
            remember(db_path, topic, fact, "fact")
            count += 1
        except ValueError:
            continue
    path.rename(path.with_name(path.name + ".imported"))
    return count
