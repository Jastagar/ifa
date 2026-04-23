from ifa.services.tts_service import TTSService
from ifa.skills.reminder import ReminderSkill
from ifa.skills.system import TimeSkill

_time_skill = TimeSkill()


def handle_with_intent(intent: str, text: str, tts: TTSService) -> str | None:
    print("DEBUG INTENT:", intent)
    if intent == "time":
        return _time_skill.handle(text)
    if intent == "reminder":
        return ReminderSkill(tts).handle(text)
    return None
