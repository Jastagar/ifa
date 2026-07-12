from ifa.core.agent_stream import agent_turn_stream
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.services.tts_service import TTSService


tts = TTSService()

ctx = AgentContext(
    db_path="ifa.db",
    tts=tts,
)

memory = Memory()


def on_sentence(sentence: str):
    print(f"\n[SENTENCE] {sentence}")


response = agent_turn_stream(
    user_text="Explain black holes simply.",
    ctx=ctx,
    memory=memory,
    on_sentence=on_sentence,
)

print("\nFINAL RESPONSE:")
print(response)