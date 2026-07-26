from dataclasses import dataclass
from ifa.skills.vision.prompt import VISION_KEYWORDS

@dataclass(slots=True)
class VisionIntent:
    requires_vision: bool

class VisionIntentDetector:

    def detect(self, text: str) -> VisionIntent:

        text = text.lower()

        return VisionIntent(
            requires_vision=any(
                keyword in text
                for keyword in VISION_KEYWORDS
            )
        )