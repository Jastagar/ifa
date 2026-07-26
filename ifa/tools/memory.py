from pathlib import Path
from datetime import datetime

from ifa.core.context import AgentContext
from ifa.tools.registry import Tool, register

MEMORY_FILE = Path("memory.md")
MAX_MEMORY_CHARS = 1000


def _ensure_memory_file():
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("# Memory\n", encoding="utf-8")


def _handler(args: dict, ctx: AgentContext) -> str:
    _ensure_memory_file()

    memory = args["memory"].strip()
    category = args.get("category", "General").strip()

    if not memory:
        return "I didn't catch what to remember."

    if len(memory) > MAX_MEMORY_CHARS:
        memory = memory[:MAX_MEMORY_CHARS]

    content = MEMORY_FILE.read_text(encoding="utf-8")

    # Prevent duplicates
    if memory.lower() in content.lower():
        return "I already know that."

    section = f"## {category}"

    if section not in content:
        content += f"\n\n{section}\n"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n- [{timestamp}] {memory}"

    section_start = content.index(section) + len(section)
    next_section = content.find("\n## ", section_start)

    if next_section == -1:
        content += entry
    else:
        content = (
            content[:next_section]
            + entry
            + content[next_section:]
        )

    MEMORY_FILE.write_text(content, encoding="utf-8")

    return "Got it, I'll remember that."


def load_memories(limit: int = 50) -> str:
    """
    Returns the contents of memory.md.
    The limit is applied to memory bullet points.
    """

    _ensure_memory_file()

    lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()

    result = []
    count = 0

    for line in lines:
        result.append(line)

        if line.startswith("- "):
            count += 1
            if count >= limit:
                break

    return "\n".join(result)


TOOL = Tool(
    name="remember",
    description=(
        "call this tool when ever user tells you to remember something, note something or check something from the memory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "memory": {
                "type": "string",
                "minLength": 1,
                "description": "The information to remember as a standalone statement.",
            },
            "category": {
                "type": "string",
                "description": "Optional category such as Personal, Preferences, Projects, Work, Health, Goals.",
                "default": "General",
            },
        },
        "required": ["memory"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)