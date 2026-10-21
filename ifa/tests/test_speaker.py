import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ifa.voice import speaker
from ifa.voice.speaker import SpeakerCheck, voiced_seconds

RATE = speaker.SAMPLE_RATE


def clip(who: int, seconds: float = 2.5, seed: int = 0) -> np.ndarray:
    audio = np.random.default_rng(seed).uniform(-0.3, 0.3, int(seconds * RATE)).astype(np.float32)
    audio[0] = who / 10
    return audio


def fake_embedder(audio: np.ndarray) -> np.ndarray:
    who = int(round(float(audio[0]) * 10))
    vector = np.zeros(8, dtype=np.float32)
    vector[who % 8] = 1.0
    vector += np.random.default_rng(len(audio)).normal(0, 0.05, 8).astype(np.float32)
    return vector / np.linalg.norm(vector)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class SpeakerCheckTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "voiceprint.json"
        self.clock = Clock()

    def make(self, **kw):
        return SpeakerCheck(profile_path=self.path, embedder=fake_embedder, threshold=0.5, trust_sec=30,
                            clock=self.clock, **kw)

    def enrolled(self):
        check = self.make()
        result = check.enroll([clip(1, seed=i) for i in range(5)], "pc")
        self.assertTrue(result.ok, result.message)
        return check

    def test_without_enrollment_everyone_is_heard(self):
        verdict = self.make().check(clip(2), "pc")
        self.assertTrue(verdict.accept)
        self.assertIn("no voice enrolled", verdict.reason)

    def test_owner_is_accepted_and_others_ignored(self):
        check = self.enrolled()
        mine, theirs = check.check(clip(1, seed=9), "pc"), check.check(clip(2, seed=9), "pc")
        self.assertTrue(mine.accept)
        self.assertGreater(mine.score, 0.9)
        self.assertFalse(theirs.accept)
        self.assertEqual(theirs.reason, "not your voice")

    def test_half_second_to_one_second_is_scored_against_the_short_threshold(self):
        check = self.enrolled()
        theirs = check.check(clip(2, seconds=0.7, seed=5), "pc")
        mine = check.check(clip(1, seconds=0.7, seed=5), "pc")
        self.assertTrue(mine.accept)
        self.assertEqual(mine.needed, check.short_threshold)
        self.assertFalse(theirs.accept)
        self.assertIn("needs 0.40", theirs.describe())

    def test_short_reply_counts_only_right_after_a_match(self):
        check = self.enrolled()
        yes = clip(2, seconds=0.4)
        self.assertFalse(check.check(yes, "pc").accept)
        check.check(clip(1, seed=3), "pc")
        self.clock.now += 10
        self.assertTrue(check.check(yes, "pc").accept)
        self.assertFalse(check.check(yes, "phone").accept)
        self.clock.now += 31
        self.assertFalse(check.check(yes, "pc").accept)

    def test_a_stranger_does_not_extend_trust(self):
        check = self.enrolled()
        check.check(clip(2, seed=4), "pc")
        self.assertFalse(check.check(clip(2, seconds=0.4), "pc").accept)

    def test_profile_is_private_and_reloads(self):
        self.enrolled()
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        again = self.make()
        self.assertEqual(again.enrolled_sources, ["pc"])
        self.assertTrue(again.check(clip(1, seed=7), "phone").accept)

    def test_sources_are_kept_separately_and_can_be_forgotten(self):
        check = self.enrolled()
        check.enroll([clip(3, seed=i) for i in range(4)], "phone")
        self.assertEqual(check.enrolled_sources, ["pc", "phone"])
        self.assertTrue(check.check(clip(3, seed=8), "pc").accept)
        check.forget("phone")
        self.assertEqual(check.enrolled_sources, ["pc"])
        check.forget()
        self.assertFalse(self.path.exists())
        self.assertTrue(check.check(clip(2), "pc").accept)

    def test_enrollment_rejects_too_little_speech_and_mixed_voices(self):
        check = self.make()
        short = check.enroll([clip(1, seconds=0.9, seed=i) for i in range(6)], "pc")
        self.assertFalse(short.ok)
        mixed = check.enroll([clip(i + 1, seed=i) for i in range(6)], "pc")
        self.assertFalse(mixed.ok)
        self.assertFalse(self.path.exists())

    def test_one_odd_recording_is_left_out(self):
        check = self.make()
        result = check.enroll([clip(1, seed=i) for i in range(5)] + [clip(4)], "pc")
        self.assertTrue(result.ok)
        self.assertEqual(result.used, 5)
        self.assertIn("left out", result.message)

    def test_profile_from_another_model_is_ignored(self):
        self.path.write_text(json.dumps({"model": "old.onnx", "sources": {"pc": {"embeddings": [[1.0] * 8]}}}))
        self.assertEqual(self.make().enrolled_sources, [])

    def test_can_be_switched_off(self):
        check = self.enrolled()
        os.environ["IFA_SPEAKER_CHECK"] = "0"
        self.addCleanup(os.environ.pop, "IFA_SPEAKER_CHECK", None)
        self.assertTrue(check.check(clip(2), "pc").accept)

    def test_voiced_seconds_ignores_silence(self):
        audio = np.concatenate([np.zeros(RATE * 2, dtype=np.float32), clip(1, seconds=1.0)])
        self.assertAlmostEqual(voiced_seconds(audio), 1.0, delta=0.05)


SR_DATA = Path(os.environ.get("IFA_SR_DATA", ""))


@unittest.skipUnless(speaker.MODEL_PATH.exists() and (SR_DATA / "enroll").is_dir(),
                     "needs the speaker model and IFA_SR_DATA pointing at sherpa-onnx's sr-data")
class RealModelTests(unittest.TestCase):
    def load(self, name):
        import wave
        with wave.open(str(next(SR_DATA.rglob(name)))) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

    def test_enrolled_speaker_is_recognised_and_others_are_not(self):
        with tempfile.TemporaryDirectory() as d:
            check = SpeakerCheck(profile_path=Path(d) / "voiceprint.json", threshold=0.5)
            result = check.enroll([self.load(f"fangjun-sr-{i}.wav") for i in (1, 2, 3)], "pc")
            self.assertTrue(result.ok, result.message)
            for name in ("fangjun-test-sr-1.wav", "fangjun-test-sr-2.wav"):
                self.assertTrue(check.check(self.load(name), "pc").accept, name)
            for name in ("leijun-test-sr-1.wav", "liudehua-test-sr-1.wav", "speaker1_a_en_16k.wav", "speaker2_a_cn_16k.wav"):
                self.assertFalse(check.check(self.load(name), "pc").accept, name)


if __name__ == "__main__":
    unittest.main()


class WakeWordShortCommandTests(unittest.TestCase):
    def test_short_command_after_the_wake_word_counts_as_you(self):
        check = SpeakerCheck(profile_path=Path(tempfile.mkdtemp()) / "vp.json", embedder=fake_embedder)
        check.enroll([clip(1, seed=s) for s in range(4)], "pc")
        stranger_short = clip(2, seconds=0.7, seed=5)
        self.assertFalse(check.check(stranger_short, "pc").accept)
        self.assertTrue(check.check(clip(2, seconds=0.4), "pc", woke=True).accept)
        self.assertFalse(check.check(clip(2, seconds=3, seed=6), "pc", woke=True).accept)
