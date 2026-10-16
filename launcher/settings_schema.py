from __future__ import annotations

from dataclasses import dataclass

TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    group: str
    kind: str = "text"
    default: str = ""
    help: str = ""
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @property
    def secret(self) -> bool:
        return self.kind == "secret"


GROUPS = (
    ("launch", "Launch"),
    ("hardware", "Hardware"),
    ("model", "Language model"),
    ("listening", "Listening"),
    ("speaking", "Speaking"),
    ("notifications", "Notifications"),
    ("keys", "Keys"),
    ("advanced", "Advanced"),
)

DEVICES = ("auto", "cuda", "cpu")

SETTINGS = (
    Setting("IFA_MODE", "Input mode", "launch", "select", "text",
            "text: type in the launcher console. voice: wake word and microphone.", ("text", "voice")),

    Setting("IFA_WHISPER_DEVICE", "Speech-to-text device", "hardware", "select", "auto",
            "Where Whisper runs. auto uses the GPU when CUDA is available.", DEVICES),
    Setting("IFA_TTS_DEVICE", "Text-to-speech device", "hardware", "select", "auto",
            "Where Chatterbox runs. auto uses the GPU when CUDA is available.", DEVICES),

    Setting("IFA_OLLAMA_MODEL", "Model", "model", "text", "",
            "Ollama model name. Required."),
    Setting("MAIN_OLLAMA_URL", "Ollama URL", "model", "text", "http://localhost:11434",
            "The single Ollama server Ifa talks to."),
    Setting("IFA_OLLAMA_THINK", "Thinking", "model", "bool", "0",
            "Let the model reason before answering. Slower replies."),
    Setting("IFA_OLLAMA_KEEP_ALIVE", "Keep model loaded", "model", "text", "10m",
            "How long Ollama keeps the model in memory between turns, e.g. 10m, 1h, 0."),

    Setting("IFA_WAKE_MODEL", "Wake word model", "listening", "text", "ifa/models/ifa.onnx",
            "Built-in name (ifa, alexa, hey_jarvis, hey_mycroft) or a path to a .onnx file."),
    Setting("IFA_WAKE_THRESHOLD", "Wake word threshold", "listening", "number", "0.7",
            "Higher means fewer false wakes but more missed ones.", minimum=0, maximum=1),
    Setting("IFA_WAKE_CONSECUTIVE", "Wake frames required", "listening", "number", "2",
            "Consecutive 80 ms frames above the threshold before waking.", minimum=1, maximum=20),
    Setting("IFA_WHISPER_MODEL", "Whisper model", "listening", "text",
            "Oriserve/Whisper-Hindi2Hinglish-Swift", "Hugging Face model id for speech-to-text."),
    Setting("IFA_VAD_SILENCE_MS", "End-of-speech silence (ms)", "listening", "number", "1500",
            "Silence that ends your turn. Lower is faster but may cut you off.", minimum=100, maximum=10000),
    Setting("IFA_VAD_MAX_UTTERANCE_MS", "Max utterance (ms)", "listening", "number", "30000",
            "Hard cap on one spoken turn.", minimum=1000, maximum=120000),
    Setting("IFA_VAD_THRESHOLD", "Speech detection threshold", "listening", "number", "0.5",
            "Silero VAD speech probability needed to count as speech.", minimum=0, maximum=1),
    Setting("IFA_FOLLOWUP_WINDOW_SEC", "Follow-up window (s)", "listening", "number", "5",
            "After a reply, listen this long without the wake word. 0 disables.", minimum=0, maximum=60),
    Setting("IFA_LISTENING_CUE", "Listening chime", "listening", "bool", "1",
            "Play a chime when Ifa starts listening."),

    Setting("IFA_VOICE_SAMPLE", "Voice sample", "speaking", "text", "ifa/audios/voice.wav",
            "WAV file Chatterbox clones Ifa's voice from."),
    Setting("IFA_FILLER_DELAY_MS", "Filler delay (ms)", "speaking", "number", "650",
            "Silence before a filler like 'One sec.' plays.", minimum=0, maximum=5000),
    Setting("IFA_FILLER_PHRASES", "Filler phrases", "speaking", "text", "",
            "Phrases Ifa says while working, separated by |. Keep them true for any request, "
            "e.g. One sec.|Just a sec. Empty uses the built-in list."),
    Setting("IFA_TTS_COOLDOWN_MS", "Echo cooldown (ms)", "speaking", "number", "500",
            "After Ifa speaks, ignore the mic this long so she doesn't hear herself.", minimum=0, maximum=5000),

    Setting("IFA_API_ENABLED", "Notification API", "notifications", "bool", "1",
            "Accept phone notifications on POST /activate."),
    Setting("IFA_API_HOST", "API host", "notifications", "text", "127.0.0.1",
            "Use 0.0.0.0 only if the bridge runs on another machine, and set a token."),
    Setting("IFA_API_PORT", "API port", "notifications", "number", "8787", "", minimum=1, maximum=65535),
    Setting("IFA_API_TOKEN", "API token", "notifications", "secret", "",
            "If set, the bridge must send it in the X-IFA-Token header."),
    Setting("IFA_NOTIF_LISTEN_SEC", "Listen after a notification (s)", "notifications", "number", "8",
            "", minimum=1, maximum=60),
    Setting("IFA_NOTIF_SETTLE_SEC", "Group messages within (s)", "notifications", "number", "4",
            "Wait this long after a sender's last message before judging the group.", minimum=0, maximum=60),
    Setting("IFA_NOTIF_MAX_RECENT", "Recent queue size", "notifications", "number", "7",
            "", minimum=1, maximum=50),
    Setting("IFA_NOTIF_TTL_SEC", "Move to discarded after (s)", "notifications", "number", "90",
            "", minimum=10, maximum=3600),
    Setting("IFA_NOTIF_MAX_DISCARDED", "Discarded queue size", "notifications", "number", "50",
            "", minimum=1, maximum=1000),

    Setting("SPOTIFY_CLIENT_ID", "Spotify client ID", "keys", "secret", "",
            "From developer.spotify.com/dashboard."),
    Setting("SPOTIFY_CLIENT_SECRET", "Spotify client secret", "keys", "secret", ""),
    Setting("SPOTIFY_REDIRECT_URI", "Spotify redirect URI", "keys", "text",
            "http://127.0.0.1:8888/callback", "Must match the URI registered in the Spotify app."),
    Setting("HF_TOKEN", "Hugging Face token", "keys", "secret", "",
            "Only needed to download gated models or avoid rate limits."),

    Setting("HF_HUB_OFFLINE", "Offline model loading", "advanced", "bool", "0",
            "Never contact Hugging Face; models must already be downloaded."),
    Setting("IFA_LATENCY_LOG", "Latency log file", "advanced", "text", "",
            "Append each turn's timing breakdown to this JSONL file."),
    Setting("IFA_VOICE_DEBUG_WAV", "Debug audio folder", "advanced", "text", "",
            "Save every captured utterance as a WAV here."),
)

BY_KEY = {s.key: s for s in SETTINGS}


def is_true(raw: str) -> bool:
    return raw.strip().lower() in TRUE_VALUES


def normalize(setting: Setting, raw: str) -> str:
    value = raw.strip()
    if setting.kind == "bool":
        return "1" if is_true(value) else "0"
    if value == "":
        return ""
    if setting.kind == "number":
        try:
            number = float(value)
        except ValueError:
            raise ValueError("must be a number") from None
        if setting.minimum is not None and number < setting.minimum:
            raise ValueError(f"must be at least {setting.minimum:g}")
        if setting.maximum is not None and number > setting.maximum:
            raise ValueError(f"must be at most {setting.maximum:g}")
        return str(int(number)) if number.is_integer() else str(number)
    if setting.kind == "select" and value not in setting.options:
        raise ValueError(f"must be one of: {', '.join(setting.options)}")
    if "\n" in value or "\r" in value:
        raise ValueError("must be a single line")
    return value
