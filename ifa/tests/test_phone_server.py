import asyncio
import json
import os
import pathlib
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from ifa.services.phone_server import (
    DISCOVERY_PROBE,
    PcmBuffer,
    PhoneServer,
    PhoneTTS,
    StreamClosed,
    has_voice,
    load_pairing_code,
    normalize,
    to_pcm16,
)


def _free_port(kind=socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _speech(seconds=1.0, amplitude=0.3) -> np.ndarray:
    t = np.arange(int(16_000 * seconds)) / 16_000
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeTTS:
    sample_rate = 24_000

    def __init__(self):
        self.calls = []

    def enqueue(self, text, sink=None):
        self.calls.append(("enqueue", text, sink))
        done = threading.Event()
        if sink is not None:
            sink(np.zeros(2400, dtype=np.float32), self.sample_rate)
        done.set()
        return done

    def play_clip(self, audio, label, sink=None):
        self.calls.append(("clip", label, sink))
        done = threading.Event()
        done.set()
        return done


class FakeSession:
    closed = False

    def __init__(self):
        self.audio_calls = []

    def send_audio(self, audio, sample_rate):
        self.audio_calls.append((len(audio), sample_rate))


class PcmBufferTests(unittest.TestCase):
    def test_read_returns_exact_chunks_across_writes(self):
        buf = PcmBuffer()
        buf.write(to_pcm16(np.full(300, 0.5, dtype=np.float32)))
        buf.write(to_pcm16(np.full(300, -0.5, dtype=np.float32)))
        first = buf.read(512)
        self.assertEqual(len(first), 512)
        self.assertAlmostEqual(float(first[0]), 0.5, places=3)
        self.assertAlmostEqual(float(first[-1]), -0.5, places=3)

    def test_close_ends_a_waiting_read(self):
        buf = PcmBuffer()
        threading.Timer(0.05, buf.close).start()
        with self.assertRaises(StreamClosed):
            buf.read(512)

    def test_stalled_stream_ends_the_read(self):
        buf = PcmBuffer(stall_sec=0.05)
        with self.assertRaises(StreamClosed):
            buf.read(512)

    def test_odd_byte_count_is_ignored_safely(self):
        buf = PcmBuffer(stall_sec=0.05)
        buf.write(b"\x01\x00\x02")
        self.assertEqual(len(buf.read(1)), 1)


class HelperTests(unittest.TestCase):
    def test_has_voice(self):
        self.assertTrue(has_voice(_speech(0.5)))
        self.assertFalse(has_voice(_speech(0.1)))
        self.assertFalse(has_voice(np.zeros(16_000, dtype=np.float32)))

    def test_normalize_boosts_quiet_audio_with_a_cap(self):
        self.assertAlmostEqual(float(np.abs(normalize(_speech(0.5, 0.05))).max()), 0.5, places=2)
        self.assertAlmostEqual(float(np.abs(normalize(_speech(0.5, 0.001))).max()), 0.02, places=3)
        loud = _speech(0.5, 0.8)
        self.assertIs(normalize(loud), loud)

    def test_pairing_code_is_generated_once_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"IFA_PHONE_TOKEN": ""}):
            path = pathlib.Path(tmp) / ".phone_token"
            code = load_pairing_code(path)
            self.assertRegex(code, r"^\d{6}$")
            self.assertEqual(load_pairing_code(path), code)

    def test_env_pairing_code_wins(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"IFA_PHONE_TOKEN": "abc"}):
            self.assertEqual(load_pairing_code(pathlib.Path(tmp) / "x"), "abc")

    def test_wrong_codes_lock_the_peer_out(self):
        server = PhoneServer(lambda *_: None, FakeTTS(), pairing_code="123456")
        for _ in range(4):
            self.assertEqual(server.check_pairing("10.0.0.2", "000000"), 401)
        self.assertEqual(server.check_pairing("10.0.0.2", "000000"), 401)
        self.assertEqual(server.check_pairing("10.0.0.2", "123456"), 429)
        self.assertEqual(server.check_pairing("10.0.0.3", "123456"), 200)

    def test_spaces_in_the_offered_code_are_ignored(self):
        server = PhoneServer(lambda *_: None, FakeTTS(), pairing_code="123456")
        self.assertEqual(server.check_pairing("10.0.0.2", " 123 456 "), 200)


class PhoneTTSTests(unittest.TestCase):
    def test_routes_speech_and_clips_to_the_phone_until_done(self):
        tts, session = FakeTTS(), FakeSession()
        phone_tts = PhoneTTS(tts, session)
        phone_tts.enqueue("hello")
        phone_tts.play_clip(np.zeros(10, dtype=np.float32), "One sec.")
        self.assertEqual(tts.calls[0][2], session.send_audio)
        self.assertEqual(tts.calls[1][2], session.send_audio)
        self.assertEqual(session.audio_calls, [(2400, 24_000)])

        phone_tts.wait_until_done()
        phone_tts.enqueue("later reminder")
        self.assertIsNone(tts.calls[-1][2])

    def test_forwards_other_attributes(self):
        phone_tts = PhoneTTS(FakeTTS(), FakeSession())
        self.assertEqual(phone_tts.sample_rate, 24_000)


class PhoneServerEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.submitted = []
        self.notifications = []

        class Inbox:
            def receive(inner, payload):
                self.notifications.append(payload)

        def submit(text, phone):
            self.submitted.append(text)
            phone.send_reply("Hi there.")
            phone.send_audio(np.zeros(9600, dtype=np.float32), 24_000)
            phone.finish_turn()

        def capture(read_chunk, start_timeout_ms=None):
            chunks = [read_chunk() for _ in range(20)]
            return np.concatenate(chunks)

        self.port = _free_port()
        self.discovery_port = _free_port(socket.SOCK_DGRAM)
        self.server = PhoneServer(
            submit, FakeTTS(), Inbox(), host="127.0.0.1", port=self.port,
            discovery_port=self.discovery_port, pairing_code="424242",
            capture=capture, transcribe=lambda audio: "turn on the lights",
            warm_up=lambda: None,
        ).start()

    def _run(self, coro):
        return asyncio.run(asyncio.wait_for(coro, timeout=10))

    def test_wrong_code_is_rejected(self):
        import aiohttp

        async def go():
            async with aiohttp.ClientSession() as http:
                with self.assertRaises(aiohttp.WSServerHandshakeError) as ctx:
                    await http.ws_connect(
                        f"http://127.0.0.1:{self.port}/phone", headers={"X-IFA-Token": "nope"}
                    )
                return ctx.exception.status

        self.assertEqual(self._run(go()), 401)

    def test_voice_turn_round_trip(self):
        import aiohttp

        async def go():
            events, audio_bytes = [], 0
            async with aiohttp.ClientSession() as http:
                ws = await http.ws_connect(
                    f"http://127.0.0.1:{self.port}/phone", headers={"X-IFA-Token": "424242"}
                )
                ready = json.loads((await ws.receive()).data)
                await ws.send_str(json.dumps({"type": "start", "startTimeoutMs": 5000}))
                pcm = to_pcm16(_speech(1.0))
                for i in range(0, len(pcm), 2560):
                    await ws.send_bytes(pcm[i:i + 2560])
                while True:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        audio_bytes += len(msg.data)
                        continue
                    event = json.loads(msg.data)
                    events.append(event)
                    if event["type"] == "turn_end":
                        break
                await ws.send_str(json.dumps({
                    "type": "notification", "packageName": "com.whatsapp",
                    "appName": "WhatsApp", "title": "Joe", "text": "free tonight?",
                }))
                await ws.send_str(json.dumps({"type": "ping"}))
                pong = json.loads((await ws.receive()).data)
                await ws.close()
            return ready, events, audio_bytes, pong

        ready, events, audio_bytes, pong = self._run(go())
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["ttsSampleRate"], 24_000)
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds, ["listening", "end_of_speech", "transcript", "reply", "audio", "turn_end"])
        self.assertEqual(events[2]["text"], "turn on the lights")
        self.assertEqual(audio_bytes, 9600 * 2)
        self.assertEqual(self.submitted, ["turn on the lights"])
        self.assertEqual(pong["type"], "pong")
        self.assertEqual(self.notifications[0]["context"]["title"], "Joe")
        self.assertEqual(self.notifications[0]["context"]["text"], "free tonight?")

    def test_silence_ends_with_no_speech(self):
        import aiohttp

        async def go():
            kinds = []
            async with aiohttp.ClientSession() as http:
                ws = await http.ws_connect(
                    f"http://127.0.0.1:{self.port}/phone", headers={"X-IFA-Token": "424242"}
                )
                await ws.receive()
                await ws.send_str(json.dumps({"type": "start"}))
                await ws.send_bytes(to_pcm16(np.zeros(16_000, dtype=np.float32)))
                while "turn_end" not in kinds:
                    kinds.append(json.loads((await ws.receive()).data)["type"])
                await ws.close()
            return kinds

        self.assertEqual(self._run(go()), ["listening", "end_of_speech", "no_speech", "turn_end"])
        self.assertEqual(self.submitted, [])

    def test_phone_stop_ends_capture_with_the_audio_so_far(self):
        import aiohttp

        def capture(read_chunk, start_timeout_ms=None):
            while True:
                read_chunk()

        self.server._capture = capture

        async def go():
            events = []
            async with aiohttp.ClientSession() as http:
                ws = await http.ws_connect(
                    f"http://127.0.0.1:{self.port}/phone", headers={"X-IFA-Token": "424242"}
                )
                await ws.receive()
                await ws.send_str(json.dumps({"type": "start"}))
                await ws.send_bytes(to_pcm16(_speech(1.0)))
                await ws.send_str(json.dumps({"type": "stop", "reason": "silence"}))
                while "turn_end" not in [e["type"] for e in events]:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        events.append(json.loads(msg.data))
                await ws.close()
            return [e["type"] for e in events]

        kinds = self._run(go())
        self.assertEqual(kinds[:3], ["listening", "end_of_speech", "transcript"])
        self.assertEqual(self.submitted, ["turn on the lights"])

    def test_discovery_answers_the_probe(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(2)
            sock.sendto(DISCOVERY_PROBE, ("127.0.0.1", self.discovery_port))
            reply = json.loads(sock.recvfrom(512)[0])
        self.assertEqual(reply["service"], "ifa")
        self.assertEqual(reply["port"], self.port)


if __name__ == "__main__":
    unittest.main()
