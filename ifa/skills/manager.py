from ifa.skills.system import TimeSkill
from ifa.skills.reminder import ReminderSkill

_time_skill = TimeSkill()


def handle_with_intent(intent: str, text: str, tts):
    print("DEBUG INTENT:", intent)
    if intent == "time":
        return _time_skill.handle(text)
    if intent == "reminder":
        return ReminderSkill(tts).handle(text)
    return None
