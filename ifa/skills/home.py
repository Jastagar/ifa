from ifa.skills.base import Skill
from ifa.skills.enums import Vibes
import sounddevice as sd
import soundfile as sf
from pathlib import Path

audio_path = Path(__file__).parent.parent / "audios" / "vibegamingmusic.wav"
data, sr = sf.read(audio_path)
class Home(Skill):
    def __init__(self, tts, default_vibe = "NORMAL"):
        self.tts = tts
        self.currentVibe:Vibes = default_vibe

    def change_vibe(self, to_vibe: Vibes) -> str:
        print("TURNING ON")
        print(to_vibe)
        if to_vibe == "GAMING":
            self.tts.speak("Sure, lets do it!")
            self.currentVibe = "GAMING"
            sd.play(data, sr)
            print("SWITCHING TO GAMING MODE")
            return 'Sure, lets do it!'
        self.currentVibe = "NORMAL"
        self.tts.speak("Switching back...")
        print("SWITCHING BACK TO NORMAL")
        return 'Switching back...'