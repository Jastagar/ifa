from PIL import Image
from ifa.skills.vision.captured import Screenshot
import mss

class ScreenCaptureService:

    def __init__(self, monitor: int = 1):
        self._sct = mss.mss()
        self._monitor = monitor

    def capture(self) -> Screenshot:
        shot = self._sct.grab(self._sct.monitors[self._monitor])

        image = Image.frombytes("RGB", shot.size, shot.rgb)

        return Screenshot(
            image=image,
            width=shot.width,
            height=shot.height,
            monitor=self._monitor,
        )

    def close(self):
        self._sct.close()