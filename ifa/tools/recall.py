import numpy as np

from ifa.core.context import AgentContext
from ifa.services import phone_server
from ifa.tools.registry import Tool

SAMPLE_RATE = 16_000
WHISPER_WINDOW = 30 * SAMPLE_RATE  # Whisper reads at most 30 s at a time


def _handler(args: dict, ctx: AgentContext) -> str:
    if phone_server.recall_seconds() == 0:
        return "ERROR: recent-audio recall is off (IFA_RECALL_SEC is 0)."
    server = phone_server.ACTIVE
    session = server.latest_session() if server else None
    if session is None:
        return "ERROR: the phone isn't connected, and only the phone keeps recent audio."
    audio = session.request_recall()
    if audio is None or len(audio) < SAMPLE_RATE:
        return "ERROR: the phone had no recent audio (it only keeps it while the app is listening)."
    from ifa.voice.stt import transcribe_array

    from ifa.services.phone_server import normalize
    parts = [transcribe_array(normalize(audio[i:i + WHISPER_WINDOW])) for i in range(0, len(audio), WHISPER_WINDOW)]
    text = " ".join(p for p in parts if p).strip()
    seconds = len(audio) / SAMPLE_RATE
    print(f"[recall] {seconds:.0f}s of recent audio: {text[:120]!r}")
    if not text:
        return f"The last {seconds:.0f} seconds had no speech the phone could make out."
    return (f"What the phone heard in the {seconds:.0f} seconds before Sir spoke (TV, other people, maybe Ifa; "
            f"not Sir's request): {text}")


TOOL = Tool(
    name="recall_recent_audio",
    description=(
        "Hear what was said around Sir in the last ~30 seconds before he spoke to you: TV, a video, a call, "
        "people nearby. Use it when he refers to something he just heard, e.g. 'is it true?', "
        "'what did he say?', 'summarise that'. For fact checks, follow with search_web. "
        "Never use it for ordinary requests."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_handler,
)
