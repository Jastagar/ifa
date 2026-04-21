import subprocess
import json
import sqlite3
from datetime import datetime

from ifa.core.memory import Memory
from ifa.services.db import DB_PATH

MODEL = "mistral"
memory = Memory()

SYSTEM = """You are Ifa, a concise and helpful assistant.
Always respond clearly in 1-2 sentences. No random text."""


# =========================
# 🧠 MAIN THINK FUNCTION
# =========================
def think(prompt: str) -> str:
    memory.add("user", prompt)

    # 🔥 build structured prompt manually
    context = ""

    # inject facts
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    facts = [row[0] for row in c.execute("SELECT fact FROM facts LIMIT 5")]
    conn.close()

    if facts:
        context += "Known facts about user:\n" + "\n".join(facts) + "\n\n"

    # inject recent memory
    for m in memory.get_recent(5):
        role = "User" if m["role"] == "user" else "Ifa"
        context += f"{role}: {m['content']}\n"

    full_prompt = SYSTEM + "\n\n" + context + "\nIfa:"

    result = subprocess.run(
        ["ollama", "run", MODEL, full_prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    response = result.stdout.strip()

    if not response:
        response = "Something went wrong."

    memory.add("assistant", response)

    return response

# =========================
# 🎯 INTENT DETECTION
# =========================
def detect_intent(text: str) -> str:
    t = text.lower()

    # 🔥 fast rules first
    if "remind" in t or "tell me to" in t:
        return "reminder"

    if "time" in t:
        return "time"

    # 🤖 fallback to LLM
    prompt = f"""
Classify into:
- time
- reminder
- none

User: {text}
Answer with one word only.
"""

    result = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    return result.stdout.strip().lower().split()[0]


# =========================
# ⏰ REMINDER EXTRACTION
# =========================
def extract_reminder(text: str) -> dict:
    prompt = f"""
Extract reminder details.

Return JSON:
- task (string)
- seconds (integer from now)

Handle:
- "in 5 seconds"
- "in 2 minutes"
- "at 2:33 am" (convert to seconds from now)

Current time: {datetime.now().strftime("%H:%M")}

User: {text}

Only return JSON.
"""

    result = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    raw = result.stdout.strip()

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {}


# =========================
# 🧠 FACT EXTRACTION
# =========================
def extract_fact(text: str) -> str | None:
    prompt = f"""
Extract a useful long-term fact from the user input.

Only return the fact.
If nothing important, return: NONE

User: {text}
"""

    result = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    fact = result.stdout.strip()

    if fact == "NONE":
        return None

    return fact