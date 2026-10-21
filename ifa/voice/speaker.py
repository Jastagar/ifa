from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{MODEL_NAME}"
MODEL_PATH = REPO_ROOT / "models" / MODEL_NAME
PROFILE_PATH = REPO_ROOT / "voiceprint.json"
SAMPLE_RATE = 16_000
MIN_VERIFY_SEC = 1.0
MIN_ENROLL_SEC = 1.2
MIN_ENROLL_SAMPLES = 3
MIN_CONSISTENCY = 0.4
SOURCES = ("pc", "phone")

ENROLL_PHRASES = [
    "Hey Ifa, what's the weather going to be like today?",
    "Play some old Arijit Singh songs on Spotify.",
    "Remind me to call mom at eight in the evening.",
    "Aaj ka plan kya hai, kuch important toh nahi hai?",
    "Open Discord and check if anyone messaged me.",
    "Theek hai, chalo phir baad mein baat karte hain.",
]


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def voiced_seconds(audio: np.ndarray, frame: int = 480) -> float:
    if len(audio) < frame:
        return 0.0
    frames = audio[: len(audio) // frame * frame].astype(np.float32).reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    floor = max(0.005, 0.1 * float(rms.max()))
    return float(np.count_nonzero(rms >= floor)) * frame / SAMPLE_RATE


def ensure_model(path: Path = MODEL_PATH, log: Callable[[str], None] = print) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[speaker] downloading the speaker model ({MODEL_NAME}, ~28 MB)...")
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


class Embedder:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        import sherpa_onnx

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=1)
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._lock = threading.Lock()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
            stream.input_finished()
            vector = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return vector / (np.linalg.norm(vector) or 1.0)


@dataclass
class Verdict:
    accept: bool
    reason: str
    score: float | None = None

    def describe(self, threshold: float) -> str:
        return self.reason + (f" (score {self.score:.2f}, needs {threshold:.2f})" if self.score is not None else "")


@dataclass
class EnrollResult:
    ok: bool
    message: str
    used: int = 0


class SpeakerCheck:
    def __init__(
        self,
        profile_path: Path = PROFILE_PATH,
        embedder: Callable[[np.ndarray], np.ndarray] | None = None,
        model_path: Path = MODEL_PATH,
        threshold: float | None = None,
        trust_sec: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.profile_path = profile_path
        self.model_path = model_path
        self.threshold = threshold if threshold is not None else float(os.environ.get("IFA_SPEAKER_THRESHOLD", "0.5"))
        self.trust_sec = trust_sec if trust_sec is not None else float(os.environ.get("IFA_SPEAKER_TRUST_SEC", "30"))
        self._embedder = embedder
        self._clock = clock
        self._lock = threading.Lock()
        self._trusted_at: dict[str, float] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._profile: dict = {}
        self.reload()

    @property
    def enrolled_sources(self) -> list[str]:
        return sorted(self._centroids)

    @property
    def active(self) -> bool:
        return _env_flag("IFA_SPEAKER_CHECK", "1") and bool(self._centroids) and self._missing() is None

    def _missing(self) -> str | None:
        if self._embedder is not None:
            return None
        if importlib.util.find_spec("sherpa_onnx") is None:
            return "sherpa-onnx is not installed (launcher: Setup, Install packages)"
        if not self.model_path.exists():
            return "the speaker model isn't downloaded (launcher: Setup, Download voice models)"
        return None

    def status(self) -> str:
        if not _env_flag("IFA_SPEAKER_CHECK", "1"):
            return "voice check off (IFA_SPEAKER_CHECK=0)"
        if not self._centroids:
            return "voice check off: no voice enrolled yet, so Ifa listens to everyone"
        missing = self._missing()
        if missing:
            return f"voice check OFF: {missing}. Ifa listens to everyone until then"
        return f"voice check on: only your voice (enrolled on {', '.join(self.enrolled_sources)}, threshold {self.threshold:.2f})"

    def reload(self) -> None:
        profile = {}
        if self.profile_path.exists():
            try:
                profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"[speaker] ignoring {self.profile_path.name}: {exc}")
        if profile and profile.get("model") != MODEL_NAME:
            print(f"[speaker] {self.profile_path.name} was made with another model; enroll your voice again")
            profile = {}
        centroids = {}
        for source, entry in (profile.get("sources") or {}).items():
            vectors = np.asarray(entry.get("embeddings") or [], dtype=np.float32)
            if vectors.ndim == 2 and len(vectors):
                mean = vectors.mean(axis=0)
                centroids[source] = mean / (np.linalg.norm(mean) or 1.0)
        with self._lock:
            self._profile, self._centroids = profile, centroids

    def embed(self, audio: np.ndarray) -> np.ndarray:
        if self._embedder is None:
            self._embedder = Embedder(self.model_path)
        return self._embedder(audio)

    def score(self, audio: np.ndarray) -> float:
        vector = self.embed(audio)
        with self._lock:
            return max(float(vector @ c) for c in self._centroids.values())

    def check(self, audio: np.ndarray, source: str) -> Verdict:
        if not self.active:
            return Verdict(True, "voice check off (no voice enrolled)" if not self._centroids else "voice check off")
        trusted = self._clock() - self._trusted_at.get(source, float("-inf")) <= self.trust_sec
        if voiced_seconds(audio) < MIN_VERIFY_SEC:
            if trusted:
                return Verdict(True, "too short to check, but you spoke a moment ago")
            return Verdict(False, "too short to tell whose voice it is")
        try:
            score = self.score(audio)
        except Exception as exc:
            print(f"[speaker] check failed, letting it through: {exc}")
            return Verdict(True, "voice check failed")
        if score >= self.threshold:
            self._trusted_at[source] = self._clock()
            return Verdict(True, "your voice", score)
        return Verdict(False, "not your voice", score)

    def enroll(self, samples: list[np.ndarray], source: str) -> EnrollResult:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        long_enough = [s for s in samples if voiced_seconds(s) >= MIN_ENROLL_SEC]
        if len(long_enough) < MIN_ENROLL_SAMPLES:
            return EnrollResult(False, f"Only {len(long_enough)} recordings had enough speech; need {MIN_ENROLL_SAMPLES}. Read each phrase fully.")
        vectors = [self.embed(s) for s in long_enough]
        kept = []
        for i, v in enumerate(vectors):
            others = np.mean([u for j, u in enumerate(vectors) if j != i], axis=0)
            similarity = float(v @ (others / (np.linalg.norm(others) or 1.0)))
            if similarity >= MIN_CONSISTENCY:
                kept.append(v)
        if len(kept) < MIN_ENROLL_SAMPLES:
            return EnrollResult(False, "The recordings didn't sound like one person. Record again somewhere quiet, just you speaking.")
        with self._lock:
            profile = dict(self._profile) or {"version": 1, "model": MODEL_NAME, "sources": {}}
            profile["model"] = MODEL_NAME
            profile.setdefault("sources", {})[source] = {
                "embeddings": [v.round(6).tolist() for v in kept],
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._write(profile)
        self.reload()
        dropped = len(long_enough) - len(kept)
        note = f" ({dropped} odd recording{'s' if dropped != 1 else ''} left out)" if dropped else ""
        return EnrollResult(True, f"Learned your voice from {len(kept)} recordings on the {source}{note}.", len(kept))

    def forget(self, source: str | None = None) -> None:
        with self._lock:
            profile = dict(self._profile)
            if source is None or not profile.get("sources"):
                profile = {}
            else:
                profile["sources"].pop(source, None)
            if profile.get("sources"):
                self._write(profile)
            elif self.profile_path.exists():
                self.profile_path.unlink()
        self.reload()

    def _write(self, profile: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.profile_path.parent, prefix=".voiceprint-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(profile, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.profile_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


_shared: SpeakerCheck | None = None
_shared_lock = threading.Lock()


def shared() -> SpeakerCheck:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = SpeakerCheck()
        return _shared
