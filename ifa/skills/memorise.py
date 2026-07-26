from pathlib import Path
from datetime import datetime

from ifa.skills.base import Skill


class MemorySkill(Skill):

    MEMORY_FILE = Path("memory.md")

    def __init__(self):
        if not self.MEMORY_FILE.exists():
            self.MEMORY_FILE.write_text("# Memory\n", encoding="utf-8")

    def save_memory(
        self,
        memory: str,
        category: str = "General",
    ) -> str:

        memory = memory.strip()

        if not memory:
            return "Cannot save an empty memory."

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        content = self.MEMORY_FILE.read_text(encoding="utf-8")

        # Avoid duplicates
        if memory.lower() in content.lower():
            return "Memory already exists."

        section = f"## {category}"

        if section not in content:
            content += f"\n\n{section}\n"

        index = content.index(section) + len(section)

        next_section = content.find("\n## ", index)

        line = f"\n- [{timestamp}] {memory}"

        if next_section == -1:
            content += line
        else:
            content = (
                content[:next_section]
                + line
                + content[next_section:]
            )

        self.MEMORY_FILE.write_text(content, encoding="utf-8")

        return "Memory saved."

    def read_memory(self) -> str:
        return self.MEMORY_FILE.read_text(encoding="utf-8")