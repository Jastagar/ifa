from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import threading
import time
from typing import Callable

import numpy as np

from ifa.services.notifications import URGENT_ANNOUNCEMENT, parse_settings
from ifa.services.pairing import lan_address, load_pairing_code

PROTOCOL_VERSION = 1
DISCOVERY_PROBE = b"IFA_DISCOVER_V1"
SAMPLE_RATE = 16_000
CAPTURE_SAMPLES = 512
AUDIO_FRAME_SAMPLES = 4800
HEARTBEAT_SEC = 30
AUDIO_STALL_SEC = 5.0
MIN_VOICE_SAMPLES = SAMPLE_RATE * 3 // 10
MIN_VOICE_PEAK = 0.01
TARGET_PEAK = 0.5
MAX_GAIN = 20.0
MAX_FAILED_PAIRINGS = 5
PAIRING_LOCKOUT_SEC = 60


def phone_enabled() -> bool:
    return os.environ.get("IFA_PHONE_ENABLED", "1").lower() not in {"0", "false", "no", "off"}


class StreamClosed(Exception):
    pass


class PcmBuffer:
    def __init__(self, stall_sec: float = AUDIO_STALL_SEC) -> None:
        self._cond = threading.Condition()
        self._samples = np.zeros(0, dtype=np.float32)
        self._closed = False
        self._close_reason = "phone stopped streaming"
        self._stall_sec = stall_sec

    def reset(self) -> None:
        with self._cond:
            self._samples = np.zeros(0, dtype=np.float32)
            self._closed = False

    def write(self, pcm16: bytes) -> None:
        usable = len(pcm16) - len(pcm16) % 2
        samples = np.frombuffer(pcm16[:usable], dtype="<i2").astype(np.float32) / 32768.0
        with self._cond:
            self._samples = np.concatenate((self._samples, samples))
            self._cond.notify_all()

    def close(self, reason: str = "phone stopped streaming") -> None:
        with self._cond:
            self._closed = True
            self._close_reason = reason
            self._cond.notify_all()

    def read(self, n: int) -> np.ndarray:
        deadline = time.monotonic() + self._stall_sec
        with self._cond:
            while len(self._samples) < n:
                if self._closed:
                    raise StreamClosed(self._close_reason)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StreamClosed("phone audio stalled")
                self._cond.wait(remaining)
            chunk, self._samples = self._samples[:n], self._samples[n:]
            return chunk


def capture_options(message: dict) -> dict:
    options = {}
    silence = message.get("silenceMs")
    if isinstance(silence, (int, float)) and not isinstance(silence, bool):
        options["silence_ms"] = int(min(max(silence, 300), 5000))
    threshold = message.get("vadThreshold")
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        options["threshold"] = float(min(max(threshold, 0.1), 0.95))
    return options


def has_voice(audio: np.ndarray) -> bool:
    return len(audio) >= MIN_VOICE_SAMPLES and float(np.abs(audio).max()) >= MIN_VOICE_PEAK


def normalize(audio: np.ndarray, target_peak: float = TARGET_PEAK, max_gain: float = MAX_GAIN) -> np.ndarray:
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak <= 0 or peak >= target_peak:
        return audio
    return audio * min(target_peak / peak, max_gain)


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class PhoneSession:
    def __init__(self, server: "PhoneServer", loop: asyncio.AbstractEventLoop, peer: str) -> None:
        self.server = server
        self.peer = peer
        self.audio = PcmBuffer()
        self.closed = False
        self._loop = loop
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._lock = threading.Lock()
        self._busy = False
        self._expect_reply = False
        self._woke = False
        self._capturing = False
        self._cancelled = False
        self._pending_nudge: str | None = None
        self._enroll_samples: list[np.ndarray] | None = None
        self._enroll_capture = False

    def send_json(self, payload: dict) -> None:
        self._post(json.dumps(payload, ensure_ascii=False))

    def send_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        pcm = to_pcm16(audio)
        self.send_json({"type": "audio", "sampleRate": int(sample_rate)})
        step = AUDIO_FRAME_SAMPLES * 2
        for start in range(0, len(pcm), step):
            self._post(pcm[start:start + step])

    def send_reply(self, text: str) -> None:
        self.send_json({"type": "reply", "text": text})

    def finish_turn(self, followup: bool = True) -> None:
        with self._lock:
            nudge, self._pending_nudge = self._pending_nudge, None
            if nudge is None:
                self._busy = False
        if nudge is not None:
            self._run_nudge(nudge)
            return
        self._expect_reply = followup and self.server.followup_sec > 0
        self.send_json({
            "type": "turn_end",
            "followup": followup and self.server.followup_sec > 0,
            "followupSec": self.server.followup_sec,
        })

    def nudge(self, level: str) -> None:
        with self._lock:
            if self._busy:
                if self._pending_nudge != "urgent":
                    self._pending_nudge = level
                return
            self._busy = True
        self._run_nudge(level)

    def greet(self, text: str, clip) -> None:
        if not self._begin():
            return

        def run() -> None:
            try:
                self.send_reply(text)
                phone_tts = PhoneTTS(self.server.tts, self)
                if clip is not None:
                    phone_tts.play_clip(clip, text)
                else:
                    phone_tts.enqueue(text)
                phone_tts.wait_until_done()
            except Exception as exc:
                print(f"[phone] couldn't play the greeting: {exc}")
            finally:
                self.finish_turn(followup=False)

        threading.Thread(target=run, name="ifa-phone-greeting", daemon=True).start()

    def _run_nudge(self, level: str) -> None:
        self.send_json({"type": "nudge", "level": level})

        def run() -> None:
            try:
                if level == "urgent":
                    phone_tts = PhoneTTS(self.server.tts, self)
                    clip = self.server.urgent_clip()
                    if clip is not None:
                        phone_tts.play_clip(clip, URGENT_ANNOUNCEMENT)
                    else:
                        phone_tts.enqueue(URGENT_ANNOUNCEMENT)
                    phone_tts.wait_until_done()
            except Exception as exc:
                print(f"[phone] couldn't play the announcement: {exc}")
            with self._lock:
                self._busy = False
            self._expect_reply = True
            self.send_json({
                "type": "turn_end",
                "followup": True,
                "followupSec": self.server.notification_listen_sec,
                "cue": True,
            })

        threading.Thread(target=run, name="ifa-phone-nudge", daemon=True).start()

    def close(self) -> None:
        self.closed = True
        self.audio.close()
        self._post(None, force=True)

    def _post(self, item, force: bool = False) -> None:
        if self.closed and not force:
            return
        try:
            self._loop.call_soon_threadsafe(self._outbox.put_nowait, item)
        except RuntimeError:
            pass

    async def run_writer(self, ws) -> None:
        while True:
            item = await self._outbox.get()
            if item is None:
                return
            if isinstance(item, bytes):
                await ws.send_bytes(item)
            else:
                await ws.send_str(item)

    def on_audio(self, data: bytes) -> None:
        if self._capturing:
            self.audio.write(data)

    def on_message(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "start":
            # Ifa didn't just ask for a reply or announce a notification, so the phone heard its wake word.
            self._woke, self._expect_reply = not self._expect_reply, False
            self._enroll_capture = bool(message.get("enroll")) and self._enroll_samples is not None
            timeout = message.get("startTimeoutMs")
            self._start_listening(int(timeout) if timeout else None, capture_options(message))
        elif kind == "stop":
            self.audio.close(f"the phone ({message.get('reason') or 'stop'})")
        elif kind == "cancel":
            self._cancelled = True
            self.audio.close("cancel")
        elif kind == "text":
            self._start_text(str(message.get("text") or "").strip())
        elif kind == "notification":
            self.server.receive_notification(message)
        elif kind == "notification_settings":
            try:
                applied = self.server.apply_notification_settings(message)
            except ValueError as exc:
                self.send_json({"type": "error", "message": f"notification settings: {exc}"})
            else:
                self.send_json({"type": "notification_settings", **applied})
        elif kind == "enroll_begin":
            self._in_thread(self._begin_enroll)
        elif kind == "enroll_finish":
            self._in_thread(self._finish_enroll)
        elif kind == "enroll_cancel":
            self._enroll_samples = None
        elif kind == "enroll_forget":
            self._in_thread(self._forget_voice)
        elif kind == "ping":
            self.send_json({"type": "pong"})
        else:
            self.send_json({"type": "error", "message": f"unknown message type {kind!r}"})

    def _begin(self) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

    def _start_listening(self, start_timeout_ms: int | None, options: dict | None = None) -> None:
        if not self._begin():
            self.send_json({"type": "error", "message": "busy"})
            return
        self.audio.reset()
        self._cancelled = False
        self._capturing = True
        self.send_json({"type": "listening"})
        threading.Thread(
            target=self._capture, args=(start_timeout_ms, options or {}), name="ifa-phone-capture", daemon=True
        ).start()

    def _start_text(self, text: str) -> None:
        if not text:
            return
        if not self._begin():
            self.send_json({"type": "error", "message": "busy"})
            return
        self.server.submit(text, self)

    def _capture(self, start_timeout_ms: int | None, options: dict) -> None:
        collected: list[np.ndarray] = []

        def read_chunk() -> np.ndarray:
            chunk = self.audio.read(CAPTURE_SAMPLES)
            collected.append(chunk)
            return chunk

        try:
            audio = self.server.capture(read_chunk, start_timeout_ms=start_timeout_ms, **options)
            ended_by = "Ifa's VAD" if len(audio) else "the start timeout (Ifa's VAD never heard speech)"
        except StreamClosed as exc:
            audio = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
            ended_by = str(exc)
        except Exception as exc:
            print(f"[phone] capture failed: {exc}")
            audio = np.zeros(0, dtype=np.float32)
            ended_by = "error"
        finally:
            self._capturing = False

        self.send_json({"type": "end_of_speech"})
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        print(
            f"[phone] captured {len(audio) / SAMPLE_RATE:.2f}s (peak {peak:.3f}), ended by {ended_by}"
        )
        self.server.dump_wav(audio)
        if self._cancelled or self.closed:
            self._no_speech("cancelled")
            return
        if self._enroll_capture:
            self._enroll_capture = False
            self._take_enroll_sample(audio)
            return
        if not has_voice(audio):
            self._no_speech(
                f"nothing loud enough to be speech (peak {peak:.3f}, ended by {ended_by})"
            )
            return

        verdict = self.server.speaker.check(audio, "phone", woke=self._woke)
        if not verdict.accept:
            self._no_speech(f"ignored, {verdict.describe(self.server.speaker.threshold)}")
            return
        if verdict.score is not None:
            print(f"[phone] {verdict.describe(self.server.speaker.threshold)}")

        print("[phone] transcribing...")
        text = (self.server.transcribe(normalize(audio)) or "").strip()
        if self._cancelled or self.closed:
            self._no_speech("cancelled")
            return
        if not text:
            self._no_speech(f"speech-to-text found no words in {len(audio) / SAMPLE_RATE:.1f}s of audio")
            return
        print(f"[phone] heard: {text!r}")
        self.send_json({"type": "transcript", "text": text})
        self.server.submit(text, self)

    def _in_thread(self, target) -> None:
        threading.Thread(target=target, name="ifa-phone-enroll", daemon=True).start()

    def _begin_enroll(self) -> None:
        from ifa.voice import speaker

        try:
            speaker.ensure_model()
        except OSError as exc:
            self.send_json({"type": "enroll_done", "ok": False, "message": f"Couldn't download the speaker model: {exc}"})
            return
        self._enroll_samples = []
        print("[phone] voice enrollment started")
        self.send_json({"type": "enroll_ready", "phrases": speaker.ENROLL_PHRASES})

    def _take_enroll_sample(self, audio: np.ndarray) -> None:
        from ifa.voice import speaker

        seconds = speaker.voiced_seconds(audio)
        samples = self._enroll_samples
        ok = samples is not None and seconds >= speaker.MIN_ENROLL_SEC
        if ok:
            samples.append(audio)
        print(f"[phone] enrollment sample: {seconds:.1f}s of speech, {'kept' if ok else 'too short'}")
        self.send_json({
            "type": "enroll_sample",
            "ok": ok,
            "count": len(samples or []),
            "seconds": round(seconds, 1),
            "message": "Got it." if ok else "Didn't catch enough speech. Say the whole phrase again.",
        })
        self.finish_turn(followup=False)

    def _finish_enroll(self) -> None:
        samples, self._enroll_samples = self._enroll_samples or [], None
        try:
            result = self.server.speaker.enroll(samples, "phone")
        except Exception as exc:
            print(f"[phone] enrollment failed: {exc}")
            self.send_json({"type": "enroll_done", "ok": False, "message": f"Enrollment failed: {exc}"})
            return
        print(f"[phone] enrollment: {result.message}")
        self.send_json({"type": "enroll_done", "ok": result.ok, "message": result.message,
                        "sources": self.server.speaker.enrolled_sources})

    def _forget_voice(self) -> None:
        self.server.speaker.forget()
        print("[phone] forgot the enrolled voice")
        self.send_json({"type": "enroll_done", "ok": True, "message": "Ifa forgot your voice. Everyone is heard again.",
                        "sources": []})

    def _no_speech(self, reason: str) -> None:
        print(f"[phone] no speech: {reason}")
        self.send_json({"type": "no_speech", "reason": reason})
        self.finish_turn(followup=False)


class PhoneTTS:
    def __init__(self, tts, session: PhoneSession) -> None:
        self._tts = tts
        self._session = session
        self._lock = threading.Lock()
        self._pending: list[threading.Event] = []
        self._open = True

    def _sink(self):
        if self._open and not self._session.closed:
            return self._session.send_audio
        return None

    def _track(self, done: threading.Event | None) -> threading.Event | None:
        if done is not None:
            with self._lock:
                self._pending.append(done)
        return done

    def enqueue(self, text: str, sink=None) -> threading.Event | None:
        return self._track(self._tts.enqueue(text, sink=sink or self._sink()))

    def speak(self, text: str) -> None:
        done = self.enqueue(text)
        if done:
            done.wait()

    def play_clip(self, audio: np.ndarray, label: str, sink=None) -> threading.Event:
        return self._track(self._tts.play_clip(audio, label, sink=sink or self._sink()))

    def wait_until_done(self) -> None:
        while True:
            with self._lock:
                pending, self._pending = self._pending, []
            if not pending:
                break
            for done in pending:
                done.wait()
        self._open = False

    def __getattr__(self, name):
        return getattr(self._tts, name)


class PhoneServer:
    def __init__(
        self,
        submit: Callable[[str, PhoneSession], None],
        tts,
        notifications=None,
        host: str | None = None,
        port: int | None = None,
        discovery_port: int | None = None,
        pairing_code: str | None = None,
        capture: Callable | None = None,
        transcribe: Callable[[np.ndarray], str] | None = None,
        warm_up: Callable[[], object] | None = None,
        urgent_clip: Callable[[], np.ndarray | None] | None = None,
        speaker_check=None,
        greeting_clip: Callable[[], np.ndarray | None] | None = None,
        greeting: str = "",
        on_greet: Callable[[], None] | None = None,
    ) -> None:
        self.submit = submit
        self.tts = tts
        self.notifications = notifications
        self.host = host or os.environ.get("IFA_PHONE_HOST", "0.0.0.0")
        self.port = port or int(os.environ.get("IFA_PHONE_PORT", "8789"))
        self.discovery_port = discovery_port or int(os.environ.get("IFA_PHONE_DISCOVERY_PORT", "8788"))
        self.pairing_code = pairing_code or load_pairing_code()
        self.followup_sec = float(os.environ.get("IFA_FOLLOWUP_WINDOW_SEC", "5"))
        self.notification_listen_sec = float(os.environ.get("IFA_NOTIF_LISTEN_SEC", "8"))
        self.priority = os.environ.get("IFA_PHONE_PRIORITY", "1").lower() not in {"0", "false", "no", "off"}
        self.urgent_clip = urgent_clip or (lambda: None)
        self.greeting = greeting
        self.on_greet = on_greet or (lambda: None)
        self.greeting_clip = greeting_clip or (lambda: None)
        self.greet_after_sec = float(os.environ.get("IFA_PHONE_GREETING_AFTER_MIN", "0")) * 60
        self._last_disconnect: float | None = None
        self._speaker = speaker_check
        self._sessions: list[PhoneSession] = []
        self._sessions_lock = threading.Lock()
        self.name = socket.gethostname()
        self._capture = capture
        self._transcribe = transcribe
        self._warm_up = warm_up
        self._warmed = threading.Event()
        self._failures: dict[str, tuple[int, float]] = {}
        self._ready = threading.Event()
        self._start_error: Exception | None = None

    @property
    def speaker(self):
        if self._speaker is None:
            from ifa.voice import speaker
            self._speaker = speaker.shared()
        return self._speaker

    def capture(self, read_chunk, start_timeout_ms: int | None = None, **options) -> np.ndarray:
        if self._capture is None:
            from ifa.voice.capture import capture_utterance
            self._capture = capture_utterance
        return self._capture(read_chunk, start_timeout_ms=start_timeout_ms, **options)

    def transcribe(self, audio: np.ndarray) -> str:
        if self._transcribe is None:
            from ifa.voice.stt import transcribe_array
            self._transcribe = transcribe_array
        return self._transcribe(audio)

    def warm_up_in_background(self) -> None:
        if self._warmed.is_set():
            return
        self._warmed.set()

        def run() -> None:
            try:
                if self._warm_up is not None:
                    self._warm_up()
                else:
                    from ifa.voice.stt import _get_model
                    _get_model()
            except Exception as exc:
                print(f"[phone] speech-to-text warm-up failed: {exc}")

        threading.Thread(target=run, name="ifa-phone-warmup", daemon=True).start()

    def has_priority(self) -> bool:
        with self._sessions_lock:
            return self.priority and bool(self._sessions)

    def nudge(self, level: str) -> bool:
        with self._sessions_lock:
            session = self._sessions[-1] if self.priority and self._sessions else None
        if session is None:
            return False
        print(f"[phone] {level} notification: nudging the phone")
        session.nudge(level)
        return True

    def receive_notification(self, message: dict) -> None:
        if self.notifications is None:
            return
        context = {k: message.get(k) for k in ("packageName", "appName", "title", "text", "receivedAt", "priority", "appNote")}
        self.notifications.receive({"context": context})

    def apply_notification_settings(self, message: dict) -> dict:
        values = parse_settings(message)
        if "listenSec" in values:
            self.notification_listen_sec = values["listenSec"]
        if self.notifications is not None:
            self.notifications.inbox.configure(
                settle_sec=values.get("settleSec"),
                ttl_sec=values.get("ttlSec"),
                max_recent=values.get("maxRecent"),
                max_discarded=values.get("maxDiscarded"),
            )
        applied = {**(self.notifications.inbox.settings() if self.notifications else {}),
                   "listenSec": self.notification_listen_sec}
        print("[phone] notification settings from the phone: " + ", ".join(f"{k}={v:g}" for k, v in applied.items()))
        return applied

    def ready_payload(self) -> dict:
        return {
            "type": "ready",
            "version": PROTOCOL_VERSION,
            "name": self.name,
            "ttsSampleRate": int(getattr(self.tts, "sample_rate", 24_000)),
            "wakeThreshold": float(os.environ.get("IFA_WAKE_THRESHOLD", "0.7")),
            "wakeConsecutive": int(os.environ.get("IFA_WAKE_CONSECUTIVE", "2")),
            "followupSec": self.followup_sec,
            "voiceSources": self.speaker.enrolled_sources,
        }

    def dump_wav(self, audio: np.ndarray) -> None:
        out_dir = os.environ.get("IFA_VOICE_DEBUG_WAV")
        if not out_dir or not len(audio):
            return
        try:
            import wave
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"ifa_phone_{int(time.time() * 1000)}.wav")
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(to_pcm16(audio))
            print(f"[phone] dumped WAV: {path}")
        except Exception as exc:
            print(f"[phone] WAV dump failed: {exc}")

    def discovery_reply(self) -> bytes:
        return json.dumps({
            "service": "ifa",
            "version": PROTOCOL_VERSION,
            "name": self.name,
            "port": self.port,
        }).encode("utf-8")

    def check_pairing(self, peer: str, offered: str) -> int:
        count, until = self._failures.get(peer, (0, 0.0))
        if until > time.monotonic():
            return 429
        offered = "".join(offered.split())
        if secrets.compare_digest(offered.encode("utf-8"), self.pairing_code.encode("utf-8")):
            self._failures.pop(peer, None)
            return 200
        count += 1
        lockout = time.monotonic() + PAIRING_LOCKOUT_SEC if count >= MAX_FAILED_PAIRINGS else 0.0
        self._failures[peer] = (0 if lockout else count, lockout)
        return 401

    def start(self) -> "PhoneServer":
        threading.Thread(target=self._serve_http, name="ifa-phone", daemon=True).start()
        self._ready.wait(timeout=10)
        if self._start_error is not None:
            raise self._start_error
        discovery = self._bind_discovery()
        if discovery is not None:
            threading.Thread(
                target=self._serve_discovery, args=(discovery,), name="ifa-phone-discovery", daemon=True
            ).start()
        print(
            f"[phone] listening on ws://{lan_address() or self.host}:{self.port}/phone "
            f"(discovery udp/{self.discovery_port}). Pairing code: {self.pairing_code}"
        )
        return self

    def _serve_http(self) -> None:
        from aiohttp import web

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(_quiet_connection_errors)
        app = web.Application()
        app.router.add_get("/phone/health", self._health)
        app.router.add_get("/phone", self._websocket)
        runner = web.AppRunner(app, access_log=None)
        try:
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, self.host, self.port).start())
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()
        loop.create_task(_watch_loop_lag())
        loop.run_forever()

    async def _health(self, request):
        from aiohttp import web
        return web.json_response({"status": "ok", "name": self.name, "version": PROTOCOL_VERSION})

    async def _websocket(self, request):
        from aiohttp import WSMsgType, web

        peer = request.remote or "unknown"
        status = self.check_pairing(peer, request.headers.get("X-IFA-Token", ""))
        if status == 429:
            return web.json_response({"error": "too many wrong pairing codes, wait a minute"}, status=429)
        if status == 401:
            offered = request.headers.get("X-IFA-Token", "")
            print(
                f"[phone] {peer} sent a wrong pairing code "
                f"({len(offered.strip())} characters{', empty' if not offered.strip() else ''}). "
                f"The code is {self.pairing_code}"
            )
            return web.json_response({"error": "wrong pairing code"}, status=401)

        # Ifa pings every 30 s and waits 15 s for the pong. Phones in Wi-Fi power saving (screen off) can hold
        # packets for several seconds, and the old 7.5 s wait dropped healthy connections.
        ws = web.WebSocketResponse(heartbeat=HEARTBEAT_SEC, max_msg_size=1 << 20)
        await ws.prepare(request)
        session = PhoneSession(self, asyncio.get_running_loop(), peer)
        writer = asyncio.create_task(session.run_writer(ws))
        session.send_json(self.ready_payload())
        self.warm_up_in_background()
        with self._sessions_lock:
            first = not self._sessions
            self._sessions.append(session)
        connected_at = time.monotonic()
        print(f"[phone] connected: {peer}" + (" (PC mic paused while the phone is connected)" if self.priority else ""))
        away = None if self._last_disconnect is None else time.monotonic() - self._last_disconnect
        if self.greeting and first and (away is None or away >= self.greet_after_sec):
            session.greet(self.greeting, self.greeting_clip())
            try:
                self.on_greet()
            except Exception as exc:
                print(f"[phone] couldn't play the greeting on the PC: {exc}")
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    session.on_audio(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        message = json.loads(msg.data)
                        if not isinstance(message, dict):
                            raise ValueError("message must be a JSON object")
                        session.on_message(message)
                    except (ValueError, TypeError) as exc:
                        session.send_json({"type": "error", "message": str(exc)})
        finally:
            with self._sessions_lock:
                self._sessions.remove(session)
                if not self._sessions:
                    self._last_disconnect = time.monotonic()
            session.close()
            try:
                await asyncio.wait_for(writer, timeout=1)
            except (asyncio.TimeoutError, Exception):
                writer.cancel()
            # Who ended it matters: a close code means a clean goodbye; 1006 or a network error means the
            # phone vanished (Wi-Fi drop, app killed, or the phone's 10 s ping got no answer).
            error = ws.exception()
            print(f"[phone] disconnected: {peer} after {time.monotonic() - connected_at:.0f}s "
                  f"(close code {ws.close_code}" + (f", {type(error).__name__}: {error}" if error else "") + ")")
        return ws

    def _bind_discovery(self) -> socket.socket | None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self.discovery_port))
            return sock
        except OSError as exc:
            print(f"[phone] discovery unavailable ({exc}); set the PC address in the app instead")
            return None

    def _serve_discovery(self, sock: socket.socket) -> None:
        reply = self.discovery_reply()
        while True:
            try:
                data, addr = sock.recvfrom(512)
                if data.strip() == DISCOVERY_PROBE:
                    sock.sendto(reply, addr)
            except OSError as exc:
                print(f"[phone] discovery error: {exc}")
                time.sleep(1)


async def _watch_loop_lag(every: float = 1.0, warn_after: float = 2.0) -> None:
    # Pongs are only sent while this loop runs. If it ever stalls, say so: that would make Ifa, not the
    # phone or Wi-Fi, the reason for "didn't receive pong" disconnects.
    loop = asyncio.get_running_loop()
    while True:
        started = loop.time()
        await asyncio.sleep(every)
        late = loop.time() - started - every
        if late > warn_after:
            print(f"[phone] the phone connection loop stalled for {late:.1f}s (pongs were delayed)")


def _quiet_connection_errors(loop, context: dict) -> None:
    # A phone vanishing mid-send leaves an unread "Connection lost" future (a long Windows traceback).
    # The disconnect line already reports it, so skip those and log everything else as usual.
    if isinstance(context.get("exception"), ConnectionError):
        return
    loop.default_exception_handler(context)


def greeting_text() -> str:
    return os.environ.get("IFA_PHONE_GREETING", "Hello sir, I am live, welcome home").strip()


def start_phone_server(submit, tts, notifications=None, urgent_clip=None, greeting_clip=None,
                       on_greet=None) -> PhoneServer | None:
    if not phone_enabled():
        print("[phone] phone app connection disabled (IFA_PHONE_ENABLED=0)")
        return None
    try:
        return PhoneServer(submit, tts, notifications, urgent_clip=urgent_clip,
                           greeting_clip=greeting_clip, greeting=greeting_text(),
                           on_greet=on_greet).start()
    except Exception as exc:
        print(f"[phone] could not start the phone server: {exc}")
        return None
