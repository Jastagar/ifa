# voice/input.py

from ifa.voice.stt import transcribe

MODE = "text"  # switch to "audio"

AUDIO_FILE = "ifa/audios/sample.wav"


def get_input():
    if MODE == "text":
        return input("You: ")
    elif MODE == "audio":
        return get_audio_input()


def get_audio_input():
    print("Listening (file mode)...")
    return transcribe(AUDIO_FILE)