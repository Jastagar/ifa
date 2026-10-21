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
    ("tools", "Tools"),
    ("notifications", "Phone app"),
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
    Setting("IFA_OLLAMA_NUM_CTX", "Context size (tokens)", "model", "number", "8192",
            "How much the model can read per request. Ifa's prompt is about 4,000 tokens before any conversation; "
            "too small and Ollama drops your message (the 'No user query found' error).", minimum=4096, maximum=65536),
    Setting("IFA_OLLAMA_KEEP_ALIVE", "Keep model loaded", "model", "text", "10m",
            "How long Ollama keeps the model in memory between turns, e.g. 10m, 1h, 0."),

    Setting("IFA_WAKE_MODEL", "Wake word model", "listening", "text", "ifa/models/ifa.onnx",
            "Built-in name (ifa, alexa, hey_jarvis, hey_mycroft) or a path to a .onnx file."),
    Setting("IFA_WAKE_THRESHOLD", "Wake word threshold", "listening", "number", "0.7",
            "Higher means fewer false wakes but more missed ones.", minimum=0, maximum=1),
    Setting("IFA_WAKE_CONSECUTIVE", "Wake frames required", "listening", "number", "2",
            "Consecutive 80 ms frames above the threshold before waking.", minimum=1, maximum=20),
    Setting("IFA_WHISPER_MODEL", "Whisper model", "listening", "text",
            "Oriserve/Whisper-Hindi2Hinglish-Apex",
            "Hugging Face model id. Oriserve/Whisper-Hindi2Hinglish-Apex: best for everyday Hinglish, fast on a GPU. "
            "-Prime: larger and slower. -Swift: tiny, for CPU only."),
    Setting("IFA_WHISPER_LANGUAGE", "Whisper language token", "listening", "select", "en",
            "The Hinglish models are trained with 'en' and write English words in English. 'hi' tends to spell them in Hindi.",
            ("en", "hi")),
    Setting("IFA_WHISPER_PROMPT", "Words to recognise", "listening", "text", "",
            "Optional names Whisper should spell right, e.g. Ifa, Spotify, YouTube, Sherlock."),
    Setting("IFA_VAD_SILENCE_MS", "End-of-speech silence (ms)", "listening", "number", "1500",
            "Silence that ends your turn. Lower is faster but may cut you off.", minimum=100, maximum=10000),
    Setting("IFA_VAD_MAX_UTTERANCE_MS", "Max utterance (ms)", "listening", "number", "30000",
            "Hard cap on one spoken turn.", minimum=1000, maximum=120000),
    Setting("IFA_VAD_THRESHOLD", "Speech detection threshold", "listening", "number", "0.5",
            "Silero VAD speech probability needed to count as speech.", minimum=0, maximum=1),
    Setting("IFA_FOLLOWUP_WINDOW_SEC", "Follow-up window (s)", "listening", "number", "5",
            "After a reply, listen this long without the wake word. 0 disables.", minimum=0, maximum=60),
    Setting("IFA_SPEAKER_CHECK", "Only listen to my voice", "listening", "bool", "1",
            "After you teach Ifa your voice (Setup on the PC, or Settings in the phone app), other voices are ignored."),
    Setting("IFA_SPEAKER_THRESHOLD", "Voice match threshold", "listening", "number", "0.5",
            "0-1. Each utterance logs its score. Lower it if Ifa ignores you; raise it if it answers others.",
            minimum=0.1, maximum=0.95),
    Setting("IFA_SPEAKER_SHORT_THRESHOLD", "Voice match threshold, short replies", "listening", "number", "0.40",
            "For 0.5-2 s of speech (\"what is it\", \"haan bolo\"). Short clips score lower, so this is below the main threshold.",
            minimum=0.1, maximum=0.95),
    Setting("IFA_SPEAKER_TRUST_SEC", "Trust short replies for (s)", "listening", "number", "90",
            "A short reply like \"yes\" can't be checked reliably; it counts as you if your voice was matched this recently. "
            "Short commands right after the wake word always count.",
            minimum=0, maximum=600),
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

    Setting("IFA_COMMANDS_ENABLED", "Allow running commands", "tools", "bool", "1",
            "Lets Ifa run terminal commands (cmd.exe on Windows). It always asks first and runs only after you say yes."),
    Setting("IFA_COMMAND_TIMEOUT_SEC", "Command time limit (s)", "tools", "number", "30",
            "Stop a command that is still running after this long.", minimum=1, maximum=600),
    Setting("IFA_NOTIF_HISTORY_DAYS", "Keep notification history (days)", "notifications", "number", "30",
            "Every notification, with the app that sent it, is saved in ifa.db (on this PC only). 0 saves nothing.",
            minimum=0, maximum=3650),
    Setting("IFA_API_ENABLED", "Notification API", "notifications", "bool", "1",
            "Accept phone notifications on POST /activate."),
    Setting("IFA_API_HOST", "API host", "notifications", "text", "127.0.0.1",
            "Use 0.0.0.0 only if the bridge runs on another machine, and set a token."),
    Setting("IFA_API_PORT", "API port", "notifications", "number", "8787", "", minimum=1, maximum=65535),
    Setting("IFA_API_TOKEN", "API token", "notifications", "secret", "",
            "If set, the bridge must send it in the X-IFA-Token header."),
    Setting("IFA_PHONE_ENABLED", "Phone app connection", "notifications", "bool", "1",
            "Let the Ifa Android app find this PC on the Wi-Fi and talk to Ifa (mic, wake word, voice). "
            "Notification settings (which apps matter, grouping, timing) live in the app: Settings, Notifications."),
    Setting("IFA_PHONE_PRIORITY", "Phone takes over when connected", "notifications", "bool", "1",
            "While the phone app is connected, the PC mic stops listening and notifications nudge the phone instead."),
    Setting("IFA_PHONE_GREETING", "Greeting when the phone connects", "notifications", "text",
            "Hello sir, I am live, welcome home",
            "Played on the phone and the PC speakers when the phone connects. Empty turns it off."),
    Setting("IFA_PHONE_GREETING_AFTER_MIN", "Greet only after being away (min)", "notifications", "number", "0",
            "0 greets on every connect. Raise it so Wi-Fi blips don't repeat the greeting.", minimum=0, maximum=1440),
    Setting("IFA_PHONE_PORT", "Phone app port", "notifications", "number", "8789",
            "WebSocket port the app connects to. Allow it through the firewall on private networks.",
            minimum=1, maximum=65535),
    Setting("IFA_PHONE_DISCOVERY_PORT", "Phone discovery port", "notifications", "number", "8788",
            "UDP port the app broadcasts to when it looks for this PC.", minimum=1, maximum=65535),
    Setting("IFA_PHONE_TOKEN", "Phone pairing code", "notifications", "secret", "",
            "The code you type into the app once. Empty: Ifa makes a 6-digit code, saves it in .phone_token and prints it at startup."),

    Setting("SPOTIFY_CLIENT_ID", "Spotify client ID", "keys", "secret", "",
            "From developer.spotify.com/dashboard."),
    Setting("SPOTIFY_CLIENT_SECRET", "Spotify client secret", "keys", "secret", ""),
    Setting("SPOTIFY_REDIRECT_URI", "Spotify redirect URI", "keys", "text",
            "http://127.0.0.1:8888/callback", "Must match the URI registered in the Spotify app."),
    Setting("IFA_GOOGLE_CREDENTIALS", "Google OAuth client file", "keys", "text", "google_credentials.json",
            "The 'Desktop app' OAuth client JSON from Google Cloud Console (Calendar API enabled). "
            "Then run Setup, Log in to Google Calendar."),
    Setting("HF_TOKEN", "Hugging Face token", "keys", "secret", "",
            "Only needed to download gated models or avoid rate limits."),

    Setting("IFA_OFFLINE", "Offline mode", "advanced", "select", "auto",
            "auto: once every model is downloaded, Ifa loads them from disk and never needs the internet. "
            "on: always offline. off: always check Hugging Face for model updates.",
            options=("auto", "on", "off")),
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
