# ifa/skills/acknowledgement/acknowledgement.py

from random import choice

from ifa.config.settings import OLLAMA_ACK_MODEL, ACK_OLLAMA_URL
from ifa.services.ollama_client import ack_chat
from ifa.skills.acknowledgement.phrases import FALLBACKS
from ifa.skills.acknowledgement.prompts import SYSTEM_PROMPT




class AcknowledgementSkill:

    def __init__(self):
        self._model = OLLAMA_ACK_MODEL

    def generate(self, user_prompt: str) -> str:
        try:
            response = ack_chat(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                think=False,
            )
            acknowledgement = (
                response.get("message", {})
                .get("content", "")
                .strip()
            )
            print("------------------")
            print("------------------")
            print(f"Acknowledgement: {acknowledgement}")
            print("------------------")
            print("------------------")
            if not acknowledgement:
                return choice(FALLBACKS)

            # Keep responses to a single short line.
            acknowledgement = " ".join(
                acknowledgement.splitlines()
            ).strip()

            # Don't let the tiny model ramble.
            if len(acknowledgement) > 80:
                acknowledgement = (
                    acknowledgement[:80]
                    .rsplit(" ", 1)[0]
                    .strip()
                )
            return acknowledgement

        except Exception as err:
            print("------------------")
            print("------------------")
            print(f"ERROR: {err}")
            print("------------------")
            print("------------------")
            return choice(FALLBACKS)