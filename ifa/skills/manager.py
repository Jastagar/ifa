from ifa.skills.system import TimeSkill
from ifa.services.tts_service import TTSService
from ifa.skills.reminder import ReminderSkill

tts = TTSService()

skills = {
    "time": TimeSkill(),
    "reminder": ReminderSkill(tts)  # ✅ THIS WAS MISSING
}

def handle_with_intent(intent: str, text: str):
    skill = skills.get(intent)
    print("DEBUG INTENT:", intent)
    if skill:
        return skill.handle(text)
    return None