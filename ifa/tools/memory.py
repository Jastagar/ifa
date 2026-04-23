"""`remember_fact` tool — LLM-initiated long-term fact storage.

Replaces the old eager `extract_fact()` per-turn pipeline. The LLM decides
when to remember by calling this tool. Facts are stored in the existing
SQLite `facts` table (INSERT OR IGNORE — duplicates are silently dropped).

`load_facts(db_path, limit)` is used by the agent's system prompt to
inject remembered facts into every turn's context (up to `limit` facts).
"""
import sqlite3

from ifa.core.context import AgentContext
from ifa.tools.registry import Tool, register

MAX_FACT_CHARS = 1000


def _handler(args: dict, ctx: AgentContext) -> str:
    fact = args["fact"].strip()
    if not fact:
        return "I didn't catch what to remember."
    if len(fact) > MAX_FACT_CHARS:
        fact = fact[:MAX_FACT_CHARS]

    try:
        conn = sqlite3.connect(ctx.db_path)
        conn.execute("INSERT OR IGNORE INTO facts (fact) VALUES (?)", (fact,))
        conn.commit()
        conn.close()
    except sqlite3.Error as exc:
        return f"I couldn't save that right now: {exc}"

    return "Got it, I'll remember that."


def load_facts(db_path: str, limit: int = 5) -> list[str]:
    """Read up to `limit` facts from the DB. Returns empty list on any error
    — the agent loop treats 'no facts' identically to 'DB unavailable'."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT fact FROM facts LIMIT ?", (limit,))
        rows = [row[0] for row in cur.fetchall()]
        conn.close()
        return rows
    except sqlite3.Error:
        return []


TOOL = Tool(
    name="remember_fact",
    description=(
        "Save a durable long-term fact about the user — names, preferences, "
        "recurring plans, relationships, anything worth recalling across "
        "conversations. Call this proactively whenever the user shares "
        "personal information worth retaining."
    ),
    parameters={
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "minLength": 1,
                "description": "The fact to remember, phrased as a standalone statement (e.g., 'user's cat is named Luna').",
            },
        },
        "required": ["fact"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)
