from ifa.skills.base import Skill

from ifa.config.settings import OLLAMA_MODEL
from ifa.services.ollama_client import chat
from ifa.skills.vision.prompt import SYSTEM_PROMPT
from ifa.skills.vision.screen_capture import ScreenCaptureService


class VisionSkill(Skill):

    def __init__(self):
        self._capture = ScreenCaptureService()
        self._model = OLLAMA_MODEL

    def analyze(self, prompt: str) -> str:
        screenshot = self._capture.capture()

        response = chat(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            images=[screenshot.image],
        )

        return response["message"]["content"]