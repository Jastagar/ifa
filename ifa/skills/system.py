from ifa.skills.base import Skill
from datetime import datetime

class TimeSkill(Skill):
    def can_handle(self, text: str) -> bool:
        return "time" in text.lower()

    def handle(self, text: str) -> str:
        return f"Current time is {datetime.now().strftime('%H:%M')}"