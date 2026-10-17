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

from ifa.services.pairing import lan_address, load_pairing_code

PROTOCOL_VERSION = 1
DISCOVERY_PROBE = b"IFA_DISCOVER_V1"
SAMPLE_RATE = 16_000
CAPTURE_SAMPLES = 512
AUDIO_FRAME_SAMPLES = 4800
AUDIO_STALL_SEC = 5.0
MIN_VOICE_SAMPLES = SAMPLE_RATE * 3 // 10
MIN_VOICE_PEAK = 0.01
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


def has_voice(audio: np.ndarray) -> bool:
    return len(audio) >= MIN_VOICE_SAMPLES and float(np.abs(audio).max()) >= MIN_VOICE_PEAK


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
        self._capturing = False
        self._cancelled = False

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
            self._busy = False
        self.send_json({
            "type": "turn_end",
            "followup": followup and self.server.followup_sec > 0,
            "followupSec": self.server.followup_sec,
        })

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
            timeout = message.get("startTimeoutMs")
            self._start_listening(int(timeout) if timeout else None)
        elif kind == "stop":
            self.audio.close(f"the phone ({message.get('reason') or 'stop'})")
        elif kind == "cancel":
            self._cancelled = True
            self.audio.close("cancel")
        elif kind == "text":
            self._start_text(str(message.get("text") or "").strip())
        elif kind == "notification":
            self.server.receive_notification(message)
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

    def _start_listening(self, start_timeout_ms: int | None) -> None:
        if not self._begin():
            self.send_json({"type": "error", "message": "busy"})
            return
        self.audio.reset()
        self._cancelled = False
        self._capturing = True
        self.send_json({"type": "listening"})
        threading.Thread(
            target=self._capture, args=(start_timeout_ms,), name="ifa-phone-capture", daemon=True
        ).start()

    def _start_text(self, text: str) -> None:
        if not text:
            return
        if not self._begin():
            self.send_json({"type": "error", "message": "busy"})
            return
        self.server.submit(text, self)

    def _capture(self, start_timeout_ms: int | None) -> None:
        collected: list[np.ndarray] = []

        def read_chunk() -> np.ndarray:
            chunk = self.audio.read(CAPTURE_SAMPLES)
            collected.append(chunk)
            return chunk

        try:
            audio = self.server.capture(read_chunk, start_timeout_ms=start_timeout_ms)
            ended_by = "Ifa's VAD"
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
        print(f"[phone] captured {len(audio) / SAMPLE_RATE:.2f}s, ended by {ended_by}")
        self.server.dump_wav(audio)
        if self._cancelled or self.closed or not has_voice(audio):
            self._no_speech()
            return

        print("[phone] transcribing...")
        text = (self.server.transcribe(audio) or "").strip()
        if not text or self._cancelled or self.closed:
            self._no_speech()
            return
        print(f"[phone] heard: {text!r}")
        self.send_json({"type": "transcript", "text": text})
        self.server.submit(text, self)

    def _no_speech(self) -> None:
        self.send_json({"type": "no_speech"})
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
    ) -> None:
        self.submit = submit
        self.tts = tts
        self.notifications = notifications
        self.host = host or os.environ.get("IFA_PHONE_HOST", "0.0.0.0")
        self.port = port or int(os.environ.get("IFA_PHONE_PORT", "8789"))
        self.discovery_port = discovery_port or int(os.environ.get("IFA_PHONE_DISCOVERY_PORT", "8788"))
        self.pairing_code = pairing_code or load_pairing_code()
        self.followup_sec = float(os.environ.get("IFA_FOLLOWUP_WINDOW_SEC", "5"))
        self.name = socket.gethostname()
        self._capture = capture
        self._transcribe = transcribe
        self._warm_up = warm_up
        self._warmed = threading.Event()
        self._failures: dict[str, tuple[int, float]] = {}
        self._ready = threading.Event()
        self._start_error: Exception | None = None

    def capture(self, read_chunk, start_timeout_ms: int | None = None) -> np.ndarray:
        if self._capture is None:
            from ifa.voice.capture import capture_utterance
            self._capture = capture_utterance
        return self._capture(read_chunk, start_timeout_ms=start_timeout_ms)

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

    def receive_notification(self, message: dict) -> None:
        if self.notifications is None:
            return
        context = {k: message.get(k) for k in ("packageName", "appName", "title", "text", "receivedAt")}
        self.notifications.receive({"context": context})

    def ready_payload(self) -> dict:
        return {
            "type": "ready",
            "version": PROTOCOL_VERSION,
            "name": self.name,
            "ttsSampleRate": int(getattr(self.tts, "sample_rate", 24_000)),
            "wakeThreshold": float(os.environ.get("IFA_WAKE_THRESHOLD", "0.7")),
            "wakeConsecutive": int(os.environ.get("IFA_WAKE_CONSECUTIVE", "2")),
            "followupSec": self.followup_sec,
            "vadSilenceMs": int(os.environ.get("IFA_VAD_SILENCE_MS", "1500")),
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

        ws = web.WebSocketResponse(heartbeat=15, max_msg_size=1 << 20)
        await ws.prepare(request)
        session = PhoneSession(self, asyncio.get_running_loop(), peer)
        writer = asyncio.create_task(session.run_writer(ws))
        session.send_json(self.ready_payload())
        self.warm_up_in_background()
        print(f"[phone] connected: {peer}")
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
            session.close()
            try:
                await asyncio.wait_for(writer, timeout=1)
            except (asyncio.TimeoutError, Exception):
                writer.cancel()
            print(f"[phone] disconnected: {peer}")
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


def start_phone_server(submit, tts, notifications=None) -> PhoneServer | None:
    if not phone_enabled():
        print("[phone] phone app connection disabled (IFA_PHONE_ENABLED=0)")
        return None
    try:
        return PhoneServer(submit, tts, notifications).start()
    except Exception as exc:
        print(f"[phone] could not start the phone server: {exc}")
        return None
