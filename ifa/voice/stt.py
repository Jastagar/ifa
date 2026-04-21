# voice/stt.py

from faster_whisper import WhisperModel

# Load once globally
model = WhisperModel("base", device="cpu", compute_type="int8")
# model = WhisperModel("tiny", device="cpu")


def transcribe(audio_path: str) -> str:
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True  # removes silence chunks
    )

    text = " ".join([segment.text for segment in segments])
    return text.strip()