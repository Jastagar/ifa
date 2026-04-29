"""Manual smoke test for Stage 2 Units 1 + 2.

Run from the repo root:

    PYTHONPATH=. venv/bin/python -m scripts.smoke_voice

What it does:

1. Exercises ``TTSService.is_speaking`` across a real speak() call.
   You should HEAR Ifa say "smoke test starting" and see the flag
   transition ``False → True → (cooldown) → False`` in the console.

2. Then runs the wake-word listener against your real mic.
   - Prints the live ``hey_jarvis`` score every ~80 ms.
   - When the score crosses the threshold, prints "*** WAKE DETECTED ***"
     and then triggers a TTS response ("yes?") to demonstrate the
     mute-during-TTS behavior.
   - Press Ctrl-C to exit.

This script is interactive and not part of the unit-test suite.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

from ifa.services.tts_service import TTSService
from ifa.voice.wake_word import (
    WAKE_CHUNK_SAMPLES,
    WakeWordInitError,
    WakeWordListener,
    _DEFAULT_MODEL,
)
# Which wake-word to probe: reads from IFA_WAKE_MODEL env var, falls
# back to the listener's _DEFAULT_MODEL (the bundled ifa.onnx) so the
# smoke harness always matches what the runtime would actually load.


SAMPLE_RATE = 16_000


def _banner(text: str) -> None:
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def smoke_unit_1(tts: TTSService) -> None:
    _banner("Unit 1 — TTSService.is_speaking flag + cooldown")
    print(f"[pre]   is_speaking = {tts.is_speaking}  (expect False)")

    observed_during = []

    def poll_during_tts() -> None:
        # Busy-ish poll for ~1s. We expect the flag to be True for most/all
        # of that window because a real TTS call takes > 1s.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            observed_during.append(tts.is_speaking)
            time.sleep(0.01)

    poller = threading.Thread(target=poll_during_tts, daemon=True)
    poller.start()
    tts.speak("smoke test starting")
    poller.join(timeout=2.0)

    true_ratio = sum(1 for x in observed_during if x) / max(len(observed_during), 1)
    print(
        f"[dur]   while speaking: {sum(observed_during)}/{len(observed_during)} "
        f"samples True ({true_ratio:.0%})"
    )
    print(f"[post]  is_speaking = {tts.is_speaking}  (expect True — in cooldown)")
    time.sleep(0.7)  # outlast default 500 ms cooldown
    print(f"[>500ms] is_speaking = {tts.is_speaking}  (expect False)")


def smoke_unit_2(tts: TTSService) -> None:
    _banner("Unit 2 — WakeWordListener")

    try:
        import sounddevice as sd  # lazy — only needed for this path
    except Exception as exc:
        print(f"[error] sounddevice import failed: {exc}")
        return

    try:
        listener = WakeWordListener(tts_service=tts)
    except WakeWordInitError as exc:
        print(f"[error] {exc}")
        return

    print(f"[info]  model = '{listener.score_key}'  threshold = {listener.threshold:.2f}")
    print("[info]  opening mic stream @ 16 kHz mono float32")
    print(f"[info]  say '{listener.score_key}' — press Ctrl-C to quit")
    print()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    stream.start()

    last_print = time.monotonic()
    peak_recent = 0.0

    def read_chunk() -> np.ndarray:
        """Read one 1280-sample chunk from the mic.

        Must NOT call ``listener._model.predict`` here — the listener's
        detect loop calls predict itself, and openWakeWord's AudioFeatures
        uses a stateful rolling buffer. Any extra predict() call advances
        that buffer out of order and destroys detection accuracy.
        Earlier versions of this script scored the chunk here for a live
        display, which dropped real-world hit rates to ~0.
        """
        nonlocal last_print, peak_recent
        data, _overflowed = stream.read(WAKE_CHUNK_SAMPLES)
        chunk = data[:, 0]

        peak_recent = max(peak_recent, float(np.abs(chunk).max()))
        now = time.monotonic()
        if now - last_print >= 0.5:
            bar_len = min(int(peak_recent * 40), 40)
            bar = "#" * bar_len + "-" * (40 - bar_len)
            mute = " [MUTED]" if tts.is_speaking else "        "
            sys.stdout.write(f"\r  mic peak: {peak_recent:.3f}  [{bar}]{mute}")
            sys.stdout.flush()
            last_print = now
            peak_recent = 0.0

        return chunk

    try:
        while True:
            score = listener.wait_for_wake(read_chunk=read_chunk)
            print(f"\n  *** WAKE DETECTED *** (score={score:.3f})")
            last_detect_at = time.monotonic()
            # Reply — this tests that (a) TTS works mid-loop, (b) the
            # listener drops frames during TTS (you should see [MUTED]
            # in the score bar while she's speaking).
            tts.speak("yes, I'm listening. keep going — say it again.")
            print("  (back to listening)")
    except KeyboardInterrupt:
        print("\n[info] exiting.")
    finally:
        stream.stop()
        stream.close()


def smoke_transcribe(tts: TTSService) -> None:
    """Full Unit 2+3+4 pipeline: wake-word → capture → Whisper transcription.

    Speak into the mic; see what Whisper thinks you said. This is the
    last smoke check before Unit 5 wires it into the orchestrator.
    """
    _banner("Unit 4 — full wake+capture+transcribe pipeline")

    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"[error] sounddevice import failed: {exc}")
        return
    try:
        listener = WakeWordListener(tts_service=tts)
    except WakeWordInitError as exc:
        print(f"[error] {exc}")
        return

    from ifa.voice.capture import CAPTURE_SAMPLES, capture_utterance
    from ifa.voice.stt import transcribe_array

    print(f"[info]  wake model = '{listener.score_key}'")
    print("[info]  whisper model loading on first transcription (lazy)...")
    print("[info]  say 'hey mycroft' then speak. Ctrl-C to exit.")
    print()

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    stream.start()

    def read_wake() -> np.ndarray:
        data, _ = stream.read(WAKE_CHUNK_SAMPLES)
        return data[:, 0]

    def read_capture() -> np.ndarray:
        data, _ = stream.read(CAPTURE_SAMPLES)
        return data[:, 0]

    try:
        while True:
            print("  listening for wake word...")
            listener.wait_for_wake(read_chunk=read_wake)
            t_wake = time.monotonic()
            print("  *** WAKE — capturing... ***")

            audio = capture_utterance(read_chunk=read_capture)
            t_captured = time.monotonic()
            dur = len(audio) / SAMPLE_RATE

            print(f"  captured {dur:.2f}s. transcribing...")
            text = transcribe_array(audio)
            t_done = time.monotonic()

            print(f"  transcribed ({t_done - t_captured:.2f}s): {text!r}")
            print(f"  total wake→done: {t_done - t_wake:.2f}s")
            print()

            tts.speak(text if text else "I didn't catch that")
            print("  (back to listening)")
            print()
    except KeyboardInterrupt:
        print("\n[info] exiting.")
    finally:
        stream.stop()
        stream.close()


def smoke_capture(tts: TTSService) -> None:
    """Drive Unit 3 (capture_utterance) against the real mic.

    Flow:
      1. Wait for wake-word (hey_mycroft)
      2. Run capture_utterance until silence / max cap
      3. Save captured audio to /tmp/ifa_capture.wav
      4. Report duration + amplitude + where the cut happened
    """
    _banner("Unit 3 — capture_utterance (say 'hey mycroft' then speak)")

    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"[error] sounddevice import failed: {exc}")
        return
    try:
        listener = WakeWordListener(tts_service=tts)
    except WakeWordInitError as exc:
        print(f"[error] {exc}")
        return

    from ifa.voice.capture import CAPTURE_SAMPLES, capture_utterance

    print(f"[info]  wake model = '{listener.score_key}'")
    print("[info]  say 'hey mycroft' then speak your message.")
    print(
        "[info]  capture ends after 1.5s silence (IFA_VAD_SILENCE_MS) "
        "or 30s (IFA_VAD_MAX_UTTERANCE_MS)."
    )
    print("[info]  Ctrl-C to exit.")
    print()

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    stream.start()

    def read_wake_chunk() -> np.ndarray:
        data, _ = stream.read(WAKE_CHUNK_SAMPLES)
        return data[:, 0]

    def read_capture_chunk() -> np.ndarray:
        data, _ = stream.read(CAPTURE_SAMPLES)
        return data[:, 0]

    try:
        while True:
            print("  listening for wake word...")
            score = listener.wait_for_wake(read_chunk=read_wake_chunk)
            t_wake = time.monotonic()
            print(f"  *** WAKE (score={score:.3f}) — now capturing utterance... ***")

            audio = capture_utterance(read_chunk=read_capture_chunk)
            elapsed = time.monotonic() - t_wake

            duration_s = len(audio) / SAMPLE_RATE
            peak = float(np.abs(audio).max()) if len(audio) else 0.0

            # Save as WAV so user can listen
            import wave
            audio_i16 = np.clip(audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
            wav_path = "/tmp/ifa_capture.wav"
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_i16.tobytes())

            print(f"    captured: {duration_s:.2f}s, peak {peak:.3f}")
            print(f"    saved:    {wav_path}  (open to verify the cut point)")
            print(f"    wall:     {elapsed:.2f}s since wake")
            print()

            tts.speak("okay, got it")
            print("  (back to listening)")
            print()
    except KeyboardInterrupt:
        print("\n[info] exiting.")
    finally:
        stream.stop()
        stream.close()


def smoke_record(duration_sec: float = 8.0) -> None:
    """Record the EXACT int16 stream fed to the wake-word model, save as WAV,
    then score it chunk-by-chunk through a fresh Model.

    Gives us two proofs in one run:
      1. The WAV lets you listen and confirm the audio is clear speech.
      2. Offline chunk-by-chunk scoring eliminates any live-stream timing
         artifacts — if the wake-word model still scores 0 on a recorded
         utterance of the target word, the problem is the model/pipeline,
         not the mic loop.
    """
    _banner(f"RECORD — capture {duration_sec:.0f}s, save WAV, offline-score")
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"[error] sounddevice import failed: {exc}")
        return

    model_name = os.environ.get("IFA_WAKE_MODEL", _DEFAULT_MODEL)
    prompt_word = (
        os.path.splitext(os.path.basename(model_name))[0].replace("_", " ")
        if os.path.exists(model_name)
        else model_name.replace("_", " ")
    )
    print(
        f"Recording {duration_sec:.0f}s. Say '{prompt_word}' 4-6 times at normal volume."
    )
    print()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    stream.start()

    chunks: list[np.ndarray] = []
    end_at = time.monotonic() + duration_sec
    last_print = time.monotonic()

    try:
        while time.monotonic() < end_at:
            data, _overflowed = stream.read(WAKE_CHUNK_SAMPLES)
            mono = data[:, 0]
            chunks.append(mono)
            now = time.monotonic()
            if now - last_print >= 0.25:
                peak = float(np.abs(mono).max())
                bar_len = min(int(peak * 40), 40)
                bar = "#" * bar_len + "-" * (40 - bar_len)
                sys.stdout.write(f"\r  live peak: {peak:.4f}  [{bar}]   ")
                sys.stdout.flush()
                last_print = now
    finally:
        stream.stop()
        stream.close()
        print()

    audio_f = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    audio_i16 = np.clip(audio_f * 32768.0, -32768.0, 32767.0).astype(np.int16)

    import wave
    import tempfile
    out_path = os.path.join(tempfile.gettempdir(), "ifa_wake_capture.wav")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_i16.tobytes())

    print()
    print(f"  saved:        {out_path} ({len(audio_i16) / SAMPLE_RATE:.2f}s)")
    print(
        f"  peak int16:   {int(np.abs(audio_i16).max())}  "
        f"(full-scale is 32767; good speech is typically 3000-15000)"
    )
    print(f"  RMS int16:    {int(np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2)))}")
    print()
    print("  STEP 1 — open the WAV and LISTEN to it.")
    print(f"    macOS: open {out_path}")
    print("    Is it clear speech? (If it's muffled / aliased / wrong voice,")
    print("    the mic pipeline is compromised and wake-word will never work.)")
    print()

    # --- offline scoring with a fresh Model ---
    try:
        from openwakeword.model import Model
        from openwakeword.utils import download_models
    except ImportError as exc:
        print(f"[error] openwakeword not installed: {exc}")
        return

    print(f"  STEP 2 — offline-scoring the recording with '{model_name}' model...")
    try:
        # Pull the ONNX if this model hasn't been cached yet
        if not os.path.exists(model_name):
            download_models(model_names=[model_name])
        model = Model(wakeword_models=[model_name], inference_framework="onnx")
    except Exception as exc:
        print(f"[error] could not load model: {exc}")
        return

    score_key = (
        os.path.splitext(os.path.basename(model_name))[0]
        if os.path.exists(model_name)
        else model_name
    )

    max_score = 0.0
    scores_over_time: list[float] = []
    for i in range(0, len(audio_i16) - WAKE_CHUNK_SAMPLES + 1, WAKE_CHUNK_SAMPLES):
        chunk = audio_i16[i : i + WAKE_CHUNK_SAMPLES]
        result = model.predict(chunk)
        s = float(result.get(score_key, 0.0))
        scores_over_time.append(s)
        max_score = max(max_score, s)

    print(f"  chunks scored:     {len(scores_over_time)}")
    print(f"  max score:         {max_score:.4f}")
    print(
        f"  99th pct:          "
        f"{np.percentile(scores_over_time, 99) if scores_over_time else 0.0:.4f}"
    )
    print(f"  >= 0.5 frames:     {sum(1 for s in scores_over_time if s >= 0.5)}")
    print(f"  >= 0.3 frames:     {sum(1 for s in scores_over_time if s >= 0.3)}")
    print()
    if max_score >= 0.5:
        print("  VERDICT: model works on the recording. Live-stream issue likely.")
    elif max_score >= 0.2:
        print("  VERDICT: model sees SOME signal. Threshold tune or better mic gain.")
    else:
        print("  VERDICT: model doesn't recognize the recording either.")
        print("    If the WAV sounded like clear speech, the model genuinely")
        print("    doesn't fit your voice. Custom training or a different built-in.")
        print("    If the WAV sounded wrong (muffled/aliased), the mic pipeline")
        print("    is the problem — investigate sample rate + device config.")


def smoke_mic(duration_sec: float = 3.0) -> None:
    """Before blaming the wake-word model, prove the mic is producing audio.

    Lists input devices, captures ``duration_sec`` from the default, and
    reports peak / RMS / mean amplitude. If these are near-zero, no wake-
    word model will ever trigger — fix the mic first.
    """
    _banner("MIC — device enumeration + amplitude check")
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"[error] sounddevice import failed: {exc}")
        return

    # --- device enumeration ---
    default_in = sd.default.device[0] if sd.default.device else None
    devices = sd.query_devices()
    print("Input devices available:")
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            marker = "  <-- default" if i == default_in else ""
            print(
                f"  [{i:2d}] {d['name']}  "
                f"({d['max_input_channels']}ch @ {int(d['default_samplerate'])}Hz){marker}"
            )
    print()

    if default_in is None:
        print("[error] no default input device detected")
        return

    default_info = devices[default_in]
    print(f"Using default: [{default_in}] {default_info['name']}")
    print()

    # --- capture ---
    print(f"Capturing {duration_sec:.0f}s at 16 kHz mono float32.")
    print("SPEAK NORMALLY during the capture window (count to five out loud).")
    print()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    stream.start()
    captured: list[np.ndarray] = []
    end_at = time.monotonic() + duration_sec
    last_print = time.monotonic()

    try:
        while time.monotonic() < end_at:
            data, _overflowed = stream.read(WAKE_CHUNK_SAMPLES)
            mono = data[:, 0]
            captured.append(mono)

            now = time.monotonic()
            if now - last_print >= 0.25:
                peak = float(np.abs(mono).max())
                bar_len = min(int(peak * 40), 40)
                bar = "#" * bar_len + "-" * (40 - bar_len)
                sys.stdout.write(f"\r  live peak: {peak:.4f}  [{bar}]   ")
                sys.stdout.flush()
                last_print = now
    finally:
        stream.stop()
        stream.close()
        print()

    arr = np.concatenate(captured) if captured else np.zeros(1)
    peak = float(np.abs(arr).max())
    rms = float(np.sqrt(np.mean(arr**2)))
    mean_abs = float(np.mean(np.abs(arr)))

    print()
    print(f"  samples:      {len(arr)}  ({len(arr) / SAMPLE_RATE:.2f} s)")
    print(f"  peak:         {peak:.4f}  (float32 range is [-1.0, 1.0])")
    print(f"  RMS:          {rms:.4f}")
    print(f"  mean |amp|:   {mean_abs:.4f}")
    print()

    if peak < 0.001:
        print("  VERDICT: mic is SILENT (peak < 0.001).")
        print("    - On macOS: System Settings → Privacy & Security → Microphone.")
        print("      Confirm your terminal (or Python) has mic access.")
        print("    - Also check System Settings → Sound → Input: is the right")
        print("      device selected, and is the input level above 0?")
    elif peak < 0.02:
        print("  VERDICT: mic signal is VERY LOW (peak < 0.02).")
        print("    Normal speech at ~20cm should peak above 0.1. Likely causes:")
        print("    - Input gain turned down in System Settings → Sound → Input.")
        print("    - Wrong input device selected (e.g., a line-in with no signal).")
        print("    - Very distant mic or heavy background environment.")
        print("    Fix the level and re-run. Wake-word scoring won't work at this level.")
    elif peak < 0.1:
        print("  VERDICT: mic signal is low but present (peak 0.02 - 0.1).")
        print("    Wake-word detection will work poorly. Raise the input gain.")
    else:
        print("  VERDICT: mic signal looks healthy (peak >= 0.1).")
        print("    If wake-word still scores near zero, the issue is not mic level.")


def smoke_diagnose(tts: TTSService, duration_sec: float = 20.0) -> None:
    """Open the mic for `duration_sec` and log per-frame wake-word scores.

    Use this when the default threshold misses too often: it tells you
    what scores a real "hey jarvis" attempt actually reaches on your
    voice + mic + room. The answer decides between tuning the threshold
    vs. switching to a different built-in model.
    """
    _banner(f"DIAGNOSE — {duration_sec:.0f}s mic capture with live scores")
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"[error] sounddevice import failed: {exc}")
        return

    try:
        listener = WakeWordListener(tts_service=tts, threshold=0.0)
    except WakeWordInitError as exc:
        print(f"[error] {exc}")
        return

    print(f"[info]  model = '{listener.score_key}'")
    print(f"[info]  say '{listener.score_key}' ~5 times, in different tones/volumes.")
    print("[info]  each bar is the max score over a 500ms window.")
    print()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    stream.start()

    all_scores: list[float] = []
    window_max = 0.0
    window_end = time.monotonic() + 0.5
    end_at = time.monotonic() + duration_sec

    try:
        while time.monotonic() < end_at:
            data, _overflowed = stream.read(WAKE_CHUNK_SAMPLES)
            chunk = data[:, 0]
            scaled = np.clip(chunk * 32768.0, -32768.0, 32767.0).astype(np.int16)
            scores = listener._model.predict(scaled)  # noqa: SLF001
            score = float(scores.get(listener.score_key, 0.0))
            all_scores.append(score)
            window_max = max(window_max, score)

            now = time.monotonic()
            if now >= window_end:
                bar_len = int(window_max * 40)
                bar = "#" * bar_len + "-" * (40 - bar_len)
                marker = "  <-- possible wake" if window_max >= 0.3 else ""
                print(f"  {window_max:.3f}  [{bar}]{marker}")
                window_max = 0.0
                window_end = now + 0.5
    finally:
        stream.stop()
        stream.close()

    # Summary
    if not all_scores:
        print("[error] no audio captured")
        return
    arr = np.array(all_scores)
    print()
    print(f"  frames scored: {len(arr)}")
    print(f"  max score:     {arr.max():.3f}")
    print(f"  99th pct:      {np.percentile(arr, 99):.3f}")
    print(f"  95th pct:      {np.percentile(arr, 95):.3f}")
    print(f"  90th pct:      {np.percentile(arr, 90):.3f}")
    print(f"  median:        {np.median(arr):.3f}")
    print()
    print("  interpretation:")
    print("    max >= 0.6 : current threshold 0.5 should work; 3/50 suggests "
          "timing or chunk-alignment glitch worth investigating")
    print("    max 0.3-0.6 : tune IFA_WAKE_THRESHOLD down to ~0.3 and retry")
    print(f"    max < 0.3  : the '{listener.score_key}' model isn't recognizing your voice "
          "— try another built-in via IFA_WAKE_MODEL, or queue up custom training")


def main() -> int:
    tts = TTSService()
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "mic":
        smoke_mic()
    elif mode == "record":
        smoke_record()
    elif mode == "capture":
        smoke_capture(tts)
    elif mode == "transcribe":
        smoke_transcribe(tts)
    elif mode == "diagnose":
        smoke_mic()
        smoke_diagnose(tts)
    elif mode == "unit1":
        smoke_unit_1(tts)
    elif mode == "unit2":
        smoke_unit_2(tts)
    else:
        smoke_unit_1(tts)
        smoke_unit_2(tts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
