from ifa.skills.base import Skill
from ifa.skills.vibe.system_prompts import __vibe_based_prompts__
from ifa.skills.enums import Vibes
import sounddevice as sd
import soundfile as sf
from pathlib import Path

audio_path = Path(__file__).parent.parent.parent / "audios" / "vibegamingmusic.wav"
data, sr = sf.read(audio_path)
class VibeManager(Skill):
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
        # if to_vibe == "GAMING":
        #     sd.play(data, sr)

        self.current_vibe = to_vibe
        return f"Vibe changed to {to_vibe} and updated system instructions are: {__vibe_based_prompts__[to_vibe]}"

    def get_current_vibe(self) -> Vibes:
        return self.current_vibe

    @staticmethod
    def get_instance():
        return VibeManager._instance

    @staticmethod
    def get_system_based_on_vibe()->str:
        return __vibe_based_prompts__[VibeManager._instance.current_vibe]