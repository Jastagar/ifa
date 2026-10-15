from ifa.skills.vibe.system_prompts import __vibe_based_prompts__
from ifa.skills.enums import Vibes


class VibeManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, tts, default_vibe="NORMAL"):
        if self._initialized:
            return

        self.tts = tts
        self.current_vibe = default_vibe
        self._initialized = True
        print(f"Default vibe is: NORMAL")

    def change_vibe(self, to_vibe: Vibes) -> str:
        self.current_vibe = to_vibe
        return f"Vibe changed to {to_vibe} and updated system instructions are: {__vibe_based_prompts__[to_vibe]}"

    @staticmethod
    def get_instance():
        return VibeManager._instance

    @staticmethod
    def get_system_based_on_vibe()->str:
        return __vibe_based_prompts__[VibeManager._instance.current_vibe]