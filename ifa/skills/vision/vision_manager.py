from ifa.skills.vision.screen_capture import ScreenCaptureService
from ifa.skills.vision import VisionService


class VisionManager:

    def __init__(
        self,
        capture: ScreenCaptureService,
        vision: VisionService,
    ):
        self._capture = capture
        self._vision = vision

    def ask(self, prompt: str) -> str:
        screenshot = self._capture.capture()

        return self._vision.analyze(
            screenshot,
            prompt,
        )