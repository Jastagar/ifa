from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

from ifa.voice import speaker


def record_phrases(source: str) -> list:
    import sounddevice as sd

    from ifa.voice.capture import CAPTURE_SAMPLES, capture_utterance

    samples = []
    with sd.InputStream(samplerate=speaker.SAMPLE_RATE, channels=1, dtype="float32") as stream:
        def read_chunk():
            data, _ = stream.read(CAPTURE_SAMPLES)
            return data[:, 0]

        for i, phrase in enumerate(speaker.ENROLL_PHRASES, 1):
            for attempt in range(3):
                print(f"\n[{i}/{len(speaker.ENROLL_PHRASES)}] Say: \"{phrase}\"", flush=True)
                time.sleep(0.3)
                if stream.read_available:
                    stream.read(stream.read_available)
                audio = capture_utterance(read_chunk, start_timeout_ms=8000, silence_ms=1000)
                seconds = speaker.voiced_seconds(audio)
                if seconds >= speaker.MIN_ENROLL_SEC:
                    print(f"    got it ({seconds:.1f}s of speech)", flush=True)
                    samples.append(audio)
                    break
                print("    didn't catch enough speech, once more please" if attempt < 2 else "    skipping this one", flush=True)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Teach Ifa your voice so it ignores everyone else.")
    parser.add_argument("--source", choices=speaker.SOURCES, default="pc")
    parser.add_argument("--forget", action="store_true", help="delete the saved voice for this source")
    args = parser.parse_args()
    load_dotenv(speaker.REPO_ROOT / ".env")

    if args.forget:
        speaker.SpeakerCheck().forget(args.source)
        print(f"Forgot your {args.source} voice.")
        return 0

    try:
        speaker.ensure_model()
    except OSError as exc:
        print(f"Couldn't download the speaker model: {exc}", file=sys.stderr)
        return 1
    print("Ifa will learn your voice from 6 short phrases. Speak normally, from where you usually sit.", flush=True)
    samples = record_phrases(args.source)
    result = speaker.SpeakerCheck().enroll(samples, args.source)
    print(("\n" if result.ok else "\nNot saved: ") + result.message, flush=True)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
