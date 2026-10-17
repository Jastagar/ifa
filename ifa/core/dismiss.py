from __future__ import annotations

import re

FILLERS = (
    "ok", "okay", "alright", "all right", "cool", "great", "nice", "perfect", "fine", "good", "hmm",
    "ifa", "sir", "boss", "for now", "achha", "acha", "theek hai", "thik hai", "haan", "so", "and", "now",
)
DISMISSALS = (
    "that's all", "thats all", "that is all", "that's it", "thats it", "that is it",
    "thank you", "thanks", "thank u", "thx", "thanks a lot", "thank you so much", "shukriya", "dhanyavaad",
    "no thanks", "no thank you", "nothing", "nothing else", "never mind", "nevermind",
    "stop listening", "go to sleep", "sleep", "cancel", "bye", "goodbye", "good bye", "bye bye", "see you",
    "bas", "bas itna hi", "bas ho gaya", "ho gaya", "kuch nahi", "chalo bye", "rehne do", "that will be all",
)
PHRASES = sorted({*FILLERS, *DISMISSALS}, key=len, reverse=True)


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z' ]+", " ", text)
    return " ".join(text.split())


def is_dismissal(text: str) -> bool:
    rest = normalize(text)
    found = False
    while rest:
        phrase = next((p for p in PHRASES if rest == p or rest.startswith(p + " ")), None)
        if phrase is None:
            return False
        found = found or phrase in DISMISSALS
        rest = rest[len(phrase):].strip()
    return found
