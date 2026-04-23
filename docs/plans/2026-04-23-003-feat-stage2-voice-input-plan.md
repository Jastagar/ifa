---
title: "feat: Stage 2 — voice input loop (wake word + VAD + STT)"
type: feat
status: active
date: 2026-04-23
origin: docs/brainstorms/2026-04-23-conversation-and-tool-foundation-requirements.md
---

# feat: Stage 2 — voice input loop (wake word + VAD + STT)

## Overview

Replace the current `input("You: ")` entry point with a voice loop: **wake word ("hey jarvis") → mic capture → VAD end-of-turn → faster-whisper STT → agent_turn → TTS response**. Text mode is preserved as a fallback and remains the default. Voice is opt-in via `IFA_MODE=voice`.

Piggybacking on this work: **Unit 6b cleanup** (the Stage 1 deletion of `brain.py` / `manager.py` / stale test stubs) lands in the same session because Stage 2 verification needs real Ollama running anyway — we gate the deletion on the Stage 1 acceptance bench at that moment.

The architectural decisions that shaped this plan are all locked upstream (see origin brainstorm + Stage 1 plan). The critical one surfaced in this session: **the mic mutes while Ifa is speaking + 500ms cooldown after TTS ends**, preventing self-trigger on her own voice. Simplest solution; zero new dependencies.

## Problem Frame

Stage 1 shipped a text-in / voice-out agent. The user's long-term vision is a voice-in / voice-out AI companion that can trigger n8n workflows. Stage 2 closes the input side: you speak, Ifa transcribes and dispatches.

Two real hazards shape the design:
1. **Self-trigger:** always-on wake-word detection on a mic that can hear Ifa's own TTS output → she wakes herself up mid-utterance. Addressed via mute-during-TTS.
2. **VAD-mediated end-of-turn:** naïve "talk until silence" cuts users off mid-thought. Addressed with a 1.5s silence threshold + 30s hard cap (tunable).

Voice I/O doesn't touch the tool-dispatch layer. Stage 1's `agent_turn(user_text, ctx, memory)` receives transcribed text and nothing downstream changes.

## Requirements Trace

Inherited from the origin brainstorm (R1-R6 under "Voice input loop" — see origin: [docs/brainstorms/2026-04-23-conversation-and-tool-foundation-requirements.md](../brainstorms/2026-04-23-conversation-and-tool-foundation-requirements.md)):

- **R1** Wake-word trigger via openWakeWord, built-in `"hey jarvis"` phrase, `inference_framework='onnx'`, one-time model download.
- **R2** Mic capture at **16kHz mono float32** until silero-vad detects ~1.5s silence; 30s max per turn.
- **R3** Captured audio → faster-whisper batch transcription (English, default `small.en`, configurable via env var).
- **R4** Transcribed text replaces `input("You: ")` in the main loop without changing the orchestrator's surface.
- **R5** After TTS playback completes, loop returns to wake-word listening (with 500ms cooldown, see R7).
- **R6** Text mode stays available as fallback via `IFA_MODE=text` (the default).

Stage-2-specific:

- **R7** `TTSService` exposes an `is_speaking` flag (set on entry to `speak()`, cleared 500ms after exit). The wake-word listener checks this flag on every frame and drops audio while set. Same mechanism handles reminder-thread TTS — reminders automatically mute wake-word listening the same way main responses do.
- **R8** Voice mode is activated by env var `IFA_MODE=voice` (default `text`). Launchers accept a flag to set this.
- **R9** When voice pipeline fails (openWakeWord models missing, mic unavailable, etc.) → fall back to text mode with a visible notice; never crash the main loop.

Unit 6b (bundled):

- **R10** Run the Stage 1 acceptance bench (`test_tool_call_bench.py`) against real Ollama. Gate the deletion commit on the bench exiting 0.
- **R11** On pass: delete `ifa/core/brain.py`, `ifa/skills/manager.py`, `ReminderSkill.handle()`, `ifa/tests/test_pipeline.py`, `ifa/tests/test_stt.py`; update `ManagerDecouplingTests` in `test_tts_service.py` to check `agent.py` imports instead of `manager.py` shape.

## Scope Boundaries

- **No custom "hey ifa" wake word** — built-in `"hey_mycroft"` ships after `hey_jarvis` (3/50) and `alexa` (0/20, 0.02 offline max) both failed on the repo owner's voice. Custom training deferred to a follow-up.
- **No Acoustic Echo Cancellation (AEC)** — mute-during-TTS is sufficient for v1. AEC is v3+ polish if ever.
- **No streaming STT** — batch transcription on end-of-turn. Latency is acceptable for conversational use (sub-second small.en + the existing LLM/TTS cadence).
- **No interruption** — user can't cut Ifa off mid-speech. Wait for TTS to finish, then trigger wake word.
- **No push-to-talk fallback** — only text mode is the fallback. PTT could be a Stage 2.5 if wake-word false-positive rate is unacceptable in practice.
- **No multi-user / voice-ID** — any voice triggering the wake word is treated as the user.

### Deferred to Separate Tasks

- **Custom "hey ifa" training** — openWakeWord's custom-model pipeline (~2-4 hours: synthetic-data generation + training run + ONNX export). User has already committed to this path once `alexa` friction grows enough. Plugging in is zero-code: drop the trained `.onnx` somewhere and set `IFA_WAKE_MODEL=/path/to/hey_ifa.onnx` — `WakeWordListener` already supports path-based specs (see `_derive_score_key`).
- **Streaming STT** — `whisper_streaming` or similar, useful if latency becomes felt. Measure before optimizing.
- **Interruption / barge-in** — explicitly accepted at v1 because user wants short, small-talk-sized replies where barge-in matters less. If replies ever grow long enough that cut-off feels useful, revisit. Two tiers of future work ready:
  - *Cheap path:* Ctrl-C keyboard handler that kills the TTS subprocess mid-utterance.
  - *Full path:* Streaming TTS + cancel (Stage 3+).
- **Bystander workflow execution mitigation** — v1 accepts the single-user assumption. If home use expands to guests or roommates, add voice-ID (speaker verification) as a guard before `call_n8n_workflow` dispatches. Plan shape: capture enrollment samples once; on each wake, compare embedding against enrolled voices; block tool calls (not all conversation) on mismatch. Off-the-shelf: `resemblyzer` or `pyannote.audio` embedding models, both offline-capable.
- **Iterative agent loop (Stage 4)** — still deferred from Stage 1; becomes more valuable as voice is in and workflows chain.

## Context & Research

### Relevant code and patterns

- [ifa/voice/input.py](../../ifa/voice/input.py) — current `MODE = "text"` switch, file-based audio stub. Target of replacement.
- [ifa/voice/stt.py](../../ifa/voice/stt.py) — current faster-whisper entry point (file-based). Becomes in-memory audio.
- [ifa/services/tts_service.py](../../ifa/services/tts_service.py) — subprocess-based TTS shipped in `d7840fe`. Gains `is_speaking` flag.
- [ifa/core/orchestrator.py](../../ifa/core/orchestrator.py) — main loop. Minimal changes: reads `IFA_MODE` once at startup, that's it (`get_input()` already abstracts over modes).
- [ifa/core/agent.py](../../ifa/core/agent.py) — unchanged. Receives transcribed text identically to typed text.
- [ifa/tests/test_tts_service.py](../../ifa/tests/test_tts_service.py) — test style to mirror.

### Institutional learnings

- The TTS commit `d7840fe` shipped a subprocess TTS that's thread-safe by construction (each `speak` spawns a fresh process). The `is_speaking` flag can wrap `subprocess.run` — set before, clear 500ms after.
- The Stage 1 test pattern (mock external deps heavily, use real SQLite + real registry for integration) carries over unchanged.

### External references

- openWakeWord 0.6.0 on PyPI. Ships ONNX models for "hey jarvis", "alexa", "hey mycroft". Must call `openwakeword.utils.download_models()` once (first run); must pass `inference_framework='onnx'` to avoid tflite-runtime dependency. License: Apache 2.0.
- faster-whisper bundles `silero_vad_v6.onnx` in its assets directory and exposes `faster_whisper.vad.SileroVADModel` backed purely by `onnxruntime`. Using this avoids a separate `silero-vad` pip install (which would pull `torch` + `torchaudio` ~600MB — see origin review for detail).
- `sounddevice.InputStream` with `samplerate=16000, dtype='float32', channels=1` matches faster-whisper's expected input directly (no resampling).

## Key Technical Decisions

- **Mute-during-TTS via `TTSService.is_speaking` flag.** `TTSService` owns the flag. `speak()` sets it on entry, clears 500ms after `subprocess.run` returns. The wake-word listener checks this on each audio frame. Same mechanism covers both the main response path and reminder daemon TTS — zero additional plumbing.
- **openWakeWord with built-in `"hey_mycroft"` for v1.** No custom training, no new external service. Model is chosen via `IFA_WAKE_MODEL` env var — accepts built-in names (`alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`) or a filesystem path to a custom `.onnx` (the forward slot for custom "hey ifa"). Selection process on the repo owner's voice: `hey_jarvis` hit 3/50 live; `alexa` hit 0/20 live and offline-max 0.02 (did not fit the voice at all); `hey_mycroft` offline-max 1.0 on an 8-second recording — adopted. Accepted under protest by the user (they dislike the "hey_" prefix). Functional > thematic until custom training lands.
- **Live wake-word listener must NOT share its Model with any side consumer.** openWakeWord's `AudioFeatures` keeps a stateful rolling buffer; every `Model.predict()` call advances it. If a debug/display consumer calls `predict()` on the same instance in parallel with the detect loop, the buffer becomes phonetically misaligned and detection accuracy collapses (hit rate went to ~0 even with a capable model during Stage 2 development). Voice wiring in Unit 5 must observe this: anything that wants live scores uses a separate `Model` instance, or extracts state via a listener-owned callback — never a second `predict()` on the listener's model.
- **Feature buffer requires continuous flow; mute feeds silence, not a gap.** During the TTS-mute window, the listener feeds int16 **silence** to `Model.predict()` every frame and discards the score. Skipping `predict()` entirely during mute (the obvious-looking approach) leaves the feature buffer frozen on the pre-mute audio — which still contains the wake-word that just fired — so the first post-mute chunk re-fires immediately, producing an infinite detection loop. Feeding silence keeps the buffer rolling in real time AND prevents Ifa's own TTS (bleeding into the mic) from contaminating features.
- **Reset prediction buffer after every successful detection.** `Model.reset()` clears the prediction buffer so the next wait_for_wake cycle starts fresh. Combined with the silence-during-mute fix, this eliminates self-retriggering.
- **silero-vad via faster-whisper's bundled ONNX.** Import `faster_whisper.vad` and use its `SileroVADModel` directly. Avoids adding `silero-vad` pip package (which pulls torch). Note: this uses faster-whisper's internal API which may change between versions — pin `faster-whisper==1.2.1` (already present) and add a version check.
- **faster-whisper `small.en` default.** 400MB model, ~300ms transcription for 5s audio on M4 Pro. Configurable via `IFA_WHISPER_MODEL=small.en` env var; ships with sensible default. English-only for v1 since user locale is English.
- **Mic capture = sounddevice InputStream, 16kHz mono float32.** Produces the exact tensor faster-whisper and silero-vad expect. Opened once at process start (not per-turn) so we don't pay device-init latency on every wake.
- **VAD end-of-turn = 1.5s sustained silence OR 30s hard cap.** Configurable via `IFA_VAD_SILENCE_MS=1500` and `IFA_VAD_MAX_UTTERANCE_MS=30000` env vars.
- **Env-var activation (`IFA_MODE=voice`).** Simpler than a config file; trivially scriptable from launchers. Default stays `text` so nothing breaks for users not opting in.
- **Graceful degradation on voice-pipeline failure.** If openWakeWord models are missing, mic is unavailable, or silero-vad init fails → print a notice and fall back to `input("You: ")` for that session. No crash.
- **Unit 6b bundled.** Since Stage 2 verification requires real Ollama running, we run the acceptance bench and execute the gated deletion in the same session. Commit 6b only lands if the bench passes.
- **Product decisions locked for v1 (confirmed 2026-04-23):**
  - "hey jarvis" ships as-is; custom "hey ifa" training deferred unless friction is felt.
  - No barge-in. The user wants short, small-talk-sized replies, which makes interruption less necessary. The system prompt in `ifa/core/agent.py` should bias toward 1-2 sentence replies — tune during Unit 5 if current behavior is too verbose.
  - Bystander-trigger risk accepted for single-user home use. Voice-ID (see Deferred) is the forward path if that changes.
  - Model-integrity hash pinning not done for v1. First-run downloads trust the HF cache.
  - All 7 env-var tunables kept — low cost, useful for diagnosing voice issues without a code edit.

## Open Questions

### Resolve Before Implementation

- **Thread topology.** The mute-during-TTS claim only holds if the TTS subprocess call and the wake-word loop run on different threads. Otherwise a synchronous `tts.speak()` blocks the wake-word loop and `is_speaking` is irrelevant. Current implicit design: wake-word loop runs on a dedicated background thread started by `VoiceInput.__init__`; the main thread blocks on a queue that the wake-word loop feeds. Before Unit 2 is written, pick one of:
  1. Wake-word runs on a background thread; `VoiceInput.get()` blocks on an internal `queue.Queue` of transcribed utterances (Recommended — keeps main thread free for TTS).
  2. Wake-word runs on the main thread and TTS is pushed to a background thread (inverts the flow; also viable but less idiomatic).
  The choice affects `is_speaking` semantics — with option 1, `tts.speak()` can block the main thread safely while the wake-word thread polls the flag. Document the decision in `ifa/voice/wake_word.py` module docstring.
- **`threading.Timer` cooldown race.** `self._cooldown_timer.cancel()` does not prevent a timer whose callback has already started running — so a rapid `speak()` after cooldown expiry can land in the wake-word's "is_speaking == False" window even though a new TTS call is about to start. Fix before Unit 1 ships: either (a) use a single re-armable `threading.Event` + monotonic expiry timestamp checked on each wake-word frame (no Timer at all — simplest), or (b) guard the Timer's clear-callback with a lock + generation counter. Option (a) is preferred; it removes the Timer class entirely.

### Resolved During Planning

- Wake-word concurrency during TTS → **mute via `is_speaking` flag + 500ms cooldown.**
- STT model → **`faster-whisper small.en`**, env-var override.
- VAD silence threshold → **1.5s**, env-var override.
- Voice mode activation → **`IFA_MODE=voice` env var.**
- Reminder TTS coordination → **same `is_speaking` flag** (free via TTSService centralization).
- Pure-ONNX silero-vad → **faster-whisper's bundled `SileroVADModel`**, no new pip dep.

### Deferred to Implementation

- **Wake-word confidence threshold.** Threshold 0.5 (openWakeWord's suggested default) was empirically too low — produced constant false positives on silence/ambient noise even with the `hey_mycroft` model that scored 1.0 on real wake words. v1 default raised to **0.7** (`IFA_WAKE_THRESHOLD`), paired with a **2-consecutive-frames-above-threshold** requirement (`IFA_WAKE_CONSECUTIVE`) to kill single-frame spikes. Latency cost: 80 ms. Both tunable.
- **Cooldown exact value.** 500ms is a starting guess. Tunable.
- **Mic device selection.** sounddevice default is the system default input. If the user's OS default is wrong (headset not detected, etc.), expose `IFA_AUDIO_INPUT_DEVICE` env var accepting a device name or index. Deferred unless it turns out to be needed.
- **openWakeWord model cache location.** Defaults to HF cache (`~/.cache/huggingface/`). Acceptable. Alternative: project-local `ifa/models/` for explicit reproducibility. Pick during implementation.
- **faster-whisper VAD internal API stability.** `faster_whisper.vad.SileroVADModel` is internal-ish. If it breaks on future faster-whisper upgrades, wrap in a try/except and fall back to a simpler energy-threshold VAD.

## Output Structure

```
ifa/
├── voice/
│   ├── input.py                  (MODIFIED — IFA_MODE env-var switch, audio branch)
│   ├── wake_word.py              (new — openWakeWord listener respecting is_speaking)
│   ├── capture.py                (new — sounddevice + VAD end-of-turn)
│   └── stt.py                    (MODIFIED — in-memory audio signature)
├── services/
│   └── tts_service.py            (MODIFIED — is_speaking flag + 500ms cooldown)
├── core/
│   └── orchestrator.py           (minor — reads IFA_MODE, logs mode at startup)
├── config/
│   └── __init__.py               (trivial — helpers for env-var parsing if useful)
└── tests/
    ├── test_wake_word.py         (new)
    ├── test_capture.py           (new)
    ├── test_stt.py               (rewritten — currently a 5-line stub)
    ├── test_voice_input.py       (new — integration: wake→capture→stt with all three mocked)
    └── test_tts_service.py       (MODIFIED — ManagerDecouplingTests rewritten for agent.py)
```

Plus in Unit 7 (delete only):
- `ifa/core/brain.py` (delete)
- `ifa/skills/manager.py` (delete)
- `ifa/skills/reminder.py` — delete `handle()` method
- `ifa/tests/test_pipeline.py` (delete — 0-byte stub)

Note: `ifa/tests/test_stt.py` is **rewritten** in Unit 4 (stub → proper tests); it is **not** deleted in Unit 7. The origin brainstorm's Unit 6b deletion list predated Unit 4's rewrite decision; the rewrite supersedes the deletion.

## High-Level Technical Design

> *Directional guidance, not implementation spec. The implementer treats it as context.*

**Voice loop (what happens after `IFA_MODE=voice`):**

```
orchestrator.run():
  startup → health check Ollama, init_db, TTSService, register_all tools
  IF IFA_MODE=voice:
    try:
      voice_pipeline = VoiceInput(tts_service)    # opens mic, loads openWakeWord + silero-vad
    except <any init failure>:
      print fallback notice; voice_pipeline = TextInput()
  ELSE:
    voice_pipeline = TextInput()

  while True:
    user_text = voice_pipeline.get()   # blocks until one complete turn captured
    if not user_text or exit-keyword: ...
    reply = agent_turn(user_text, ctx, memory)
    tts.speak(reply)   # sets is_speaking=True, subprocess runs, clears 500ms later
```

**`VoiceInput.get()` internals:**

```
1. Wait for wake word:
   WHILE True:
     frame = mic.read(chunk_size)
     if tts.is_speaking: continue
     score = wakeword_model.predict(frame)
     if score >= threshold:
       play a soft "listening" cue (optional); break

2. Capture utterance:
   audio_buffer = []
   silence_ms = 0
   start = time.monotonic()
   WHILE True:
     frame = mic.read(chunk_size)
     audio_buffer.append(frame)
     if vad.is_speech(frame):
       silence_ms = 0
     else:
       silence_ms += chunk_duration_ms
       if silence_ms >= SILENCE_THRESHOLD: break
     if time.monotonic() - start >= MAX_UTTERANCE_SEC: break

3. Transcribe + return:
   text = whisper.transcribe(concat(audio_buffer))
   return text
```

**TTS mute mechanism** (Timer-free; see Unit 1 Approach for rationale):

```
class TTSService:
  _lock = threading.Lock()
  _active_count = 0           # in-flight speak() calls
  _mute_until = 0.0           # monotonic timestamp; mic stays muted until now >= this

  @property
  def is_speaking(self) -> bool:
    with self._lock:
      return self._active_count > 0 or time.monotonic() < self._mute_until

  def speak(self, text):
    if not text: return
    with self._lock:
      self._active_count += 1
    try:
      subprocess.run(...)
    finally:
      with self._lock:
        self._active_count -= 1
        self._mute_until = time.monotonic() + 0.5
```

## Implementation Units

- [ ] **Unit 1: TTSService `is_speaking` flag + cooldown**

**Goal:** Add the mute-state mechanism that wake-word listener will check. Zero behavior change in text mode.

**Requirements:** R7

**Dependencies:** None

**Files:**
- Modify: `ifa/services/tts_service.py` — add `_is_speaking` threading.Event, `is_speaking` property, cooldown Timer; flip in `speak()`.
- Modify: `ifa/tests/test_tts_service.py` — add tests for `is_speaking` transitions.

**Approach:**
- Owned state on TTSService: `_speaking_lock = threading.Lock()`, `_active_count = 0` (int, guards reentrancy), `_mute_until = 0.0` (monotonic timestamp).
- `is_speaking` property (read under `_speaking_lock`):
  ```
  return self._active_count > 0 or time.monotonic() < self._mute_until
  ```
  Returning True either while a subprocess call is in flight *or* during the cooldown window. Readers (wake-word loop on another thread) hit the lock only for a few nanoseconds.
- `speak()` increments `_active_count` on entry under the lock; in `finally` decrements it and sets `self._mute_until = time.monotonic() + 0.5`. No `threading.Timer`, no `cancel()` race: overlapping `speak()` calls simply extend the window because each sets a fresh `_mute_until` and the check is "now < mute_until".
- Text-mode callers never observe the flag; voice-mode will.
- This approach was chosen over `threading.Event + Timer` because `Timer.cancel()` does not abort a callback that has already begun running — under rapid back-to-back `speak()` calls, the old timer's `clear()` callback could fire *after* a new `speak()` has re-set the event, opening a brief "mute-off" window the wake-word loop could exploit to trigger on Ifa's own voice. The monotonic-timestamp approach has no such window. (See Open Questions → Resolve Before Implementation.)

**Test scenarios:**
- Happy path: `is_speaking` False before `speak()`, True during subprocess call, True during the 500ms cooldown, False once `time.monotonic() >= _mute_until`.
- Edge case: two `speak()` calls within 500ms — `is_speaking` stays True continuously across both calls and the trailing cooldown (no premature clear).
- Edge case (**Timer-race regression guard**): rapid `speak()` → returns → `speak()` again, simulated with a patched `time.monotonic`. Prove there is no window where `is_speaking == False` between the two calls. This is the specific race the monotonic-timestamp design eliminates vs. the Timer-based alternative.
- Edge case: reminder daemon thread calling `speak()` in parallel with main-thread `speak()` — `_active_count` tracks both; `is_speaking` True while either is active.
- Integration: existing `speak()` behavior (audible output, error handling) unchanged. All TTSService tests from commit `d7840fe` continue to pass.

**Verification:** All existing TTSService tests still pass; new state-transition tests pass.

---

- [ ] **Unit 2: Wake-word listener**

**Goal:** Continuous audio loop detecting "hey jarvis" via openWakeWord, respecting `tts.is_speaking`.

**Requirements:** R1, R7 (consumer side)

**Dependencies:** Unit 1 (is_speaking flag), openWakeWord dep

**Files:**
- Create: `ifa/voice/wake_word.py` — `WakeWordListener` class with `wait_for_wake(mic_stream) -> None`
- Modify: `ifa/requirements.txt` — add `openwakeword==0.6.0`. openWakeWord transitively pulls `numpy`, `scipy`, `scikit-learn`, `onnxruntime` — after first install, run `pip freeze | grep -E 'openwakeword|scipy|scikit-learn|onnxruntime'` and add those pinned versions to `requirements.txt` so subsequent installs are reproducible.
- Create: `ifa/tests/test_wake_word.py`

**Approach:**
- On `__init__`: call `openwakeword.utils.download_models(model_names=['hey_jarvis'])` (first-run download; subsequent calls are no-ops — the util checks the target file's existence before re-downloading). Instantiate `openwakeword.model.Model(wakeword_models=['hey_jarvis'], inference_framework='onnx')`.
- Note: the *pretrained name* used by openWakeWord's public API is the short form `'hey_jarvis'`. The library internally maps this to the file `hey_jarvis_v0.1.onnx` (or `.tflite`) but callers never see the versioned filename — `Model.predict()` returns a dict keyed by the short name: `{'hey_jarvis': 0.0-1.0}`.
- `wait_for_wake(read_chunk, tts)` loops calling the injected `read_chunk()` callable which returns 1280-sample frames (80ms at 16kHz, openWakeWord's expected input). In production, `read_chunk` wraps `stream.read(1280)` and extracts the 1D mono array via `data[:, 0]`. In tests, `read_chunk` is a mock. If `tts.is_speaking`, continue without scoring. Otherwise call `model.predict(chunk_i16)` — openWakeWord expects int16 PCM audio; sounddevice produces float32 in `[-1.0, 1.0]`, so convert with `(chunk * 32768).clip(-32768, 32767).astype(np.int16)`. Check if `scores['hey_jarvis'] >= threshold` (from `IFA_WAKE_THRESHOLD` env, default 0.5). Return when hit.
- Threshold, chunk size, model name all module-level constants for easy tuning.
- Note on shared stream: the `sounddevice.InputStream` is opened once by `VoiceInput` (Unit 5). Different phases of the voice loop request different frame sizes from the same stream — wake-word reads 1280-sample frames; capture (Unit 3) reads 512-sample frames. `InputStream.read(n)` honors the requested n regardless of the stream's `blocksize` hint at open time, so a single long-lived stream supports both consumers without reopening. Only one consumer reads at a time (sequential: wait-for-wake → capture → transcribe). The listener receives `read_chunk` as a callable rather than the raw stream object to keep the unit testable without sounddevice.

**Test scenarios:**
- Happy path: mocked model returns high score → `wait_for_wake` returns.
- Edge case: `tts.is_speaking=True` → audio frames are skipped (model.predict not called). Unmute → frames resume.
- Error path: `download_models()` raises → constructor raises a specific exception with actionable message.
- Integration: construct listener with a real openWakeWord but mocked mic stream (fake audio) → verify end-to-end without real audio.

**Verification:** Listener detects a mocked-high-score frame; ignores audio during TTS mute window.

---

- [ ] **Unit 3: Mic capture + VAD end-of-turn**

**Goal:** After wake-word fires, capture audio until 1.5s silence (or 30s cap).

**Requirements:** R2

**Dependencies:** Unit 2 (same mic stream), faster-whisper's `SileroVADModel`

**Files:**
- Create: `ifa/voice/capture.py` — `capture_utterance(mic_stream) -> np.ndarray`
- Create: `ifa/tests/test_capture.py`

**Approach:**
- Read from an already-open `sounddevice.InputStream` (opened once at process start; passed in). `stream.read(512)` returns `(data, overflowed)` where data is shape `(512, 1)` float32; flatten via `data[:, 0]` to the 1D array VAD expects.
- Each 512-sample chunk (32ms at 16kHz) is scored by calling the `SileroVADModel` instance directly: `prob = vad_model(chunk_1d, 16000)` — `SileroVADModel.__call__` is the public way to get a speech-probability float; there is no separate `get_speech_probability` method. This is an internal-ish API of `faster_whisper.vad` — wrap the call in try/except with an energy-threshold fallback (RMS > threshold → treat as speech) if the model raises or the attribute disappears on upgrade.
- If probability >= 0.5, reset silence counter. Else increment by chunk duration. If counter >= `IFA_VAD_SILENCE_MS` (default 1500), break.
- Hard cap: `IFA_VAD_MAX_UTTERANCE_MS` (default 30000).
- Return the concatenated float32 numpy array (entire utterance including trailing silence, trimmed at the cap).

**Test scenarios:**
- Happy path: mocked VAD returns speech=True for N chunks, then speech=False for 1.5s → capture returns array of length N+silence-equivalent.
- Edge case: VAD never detects silence → capture returns at 30s cap (verifies max utterance).
- Edge case: VAD raises exception → falls back to energy-threshold VAD; capture still completes.
- Edge case: Empty audio (immediate silence after wake word) → returns short array or empty; STT handles the empty-text case gracefully.

**Verification:** 10-second synthetic audio with 2s speech + 2s silence → capture returns ~3.5s of audio (speech + 1.5s silence).

---

- [ ] **Unit 4: In-memory STT**

**Goal:** Run faster-whisper on a numpy array (not a file path) and return transcribed text.

**Requirements:** R3

**Dependencies:** faster-whisper (already present), Unit 3 output (numpy float32 @ 16kHz)

**Files:**
- Modify: `ifa/voice/stt.py` — currently file-based. Add `transcribe_array(audio: np.ndarray) -> str`.
- Modify: `ifa/tests/test_stt.py` — currently a 5-line stub; rewrite as proper unit tests.

**Approach:**
- Initialize `WhisperModel(os.environ.get('IFA_WHISPER_MODEL', 'small.en'), compute_type='int8')` once at process start; re-use across turns. Note: the first constructor argument is `model_size_or_path` (positional) — there is no keyword `size=`. int8 quantization is small.en-appropriate and fits comfortably on any target device.
- `transcribe_array(audio)` calls `model.transcribe(audio, language='en')` where `audio` is a 1D float32 numpy array at 16kHz (what Units 3 and 5 produce). The call returns `(segments, info)` — iterate `segments` and concatenate each `segment.text`, strip whitespace, return.
- Handle `RuntimeError` during transcription by returning empty string (the agent loop treats empty input as a no-op turn).

**Test scenarios:**
- Happy path: mocked Whisper returns known segments → transcribe_array returns concatenated text.
- Edge case: empty audio array → empty string.
- Error path: Whisper raises RuntimeError mid-transcription → returns empty string, does not raise.
- Integration: real Whisper on a tiny generated sine-wave → returns *something* (even if it's gibberish) without crashing. Proves the init + inference path.

**Verification:** Text tests pass; can import and construct the model without exceptions on M4 Pro and CUDA Linux.

---

- [ ] **Unit 5: Voice input wiring**

**Goal:** `voice/input.py` routes to the voice pipeline when `IFA_MODE=voice`; falls back to text on any init failure.

**Requirements:** R4, R6, R8, R9

**Dependencies:** Units 1-4

**Files:**
- Modify: `ifa/voice/input.py` — class `VoiceInput` that opens mic stream once, holds WakeWordListener + VAD + Whisper; `get()` method returns one utterance's text.
- Modify: `ifa/core/orchestrator.py` — read `IFA_MODE` env at startup; construct `VoiceInput` or keep existing `get_input` function; log mode clearly.
- Create: `ifa/tests/test_voice_input.py` — integration with all external deps mocked.

**Approach:**
- `VoiceInput.__init__(tts_service)`: opens sounddevice.InputStream once; constructs WakeWordListener, capture helper, STT model.
- `VoiceInput.get() -> str`: waits for wake word; captures utterance; transcribes; returns text.
- `VoiceInput.close()`: stops the mic stream (called on orchestrator exit).
- Orchestrator's `run()`:
  ```
  mode = os.environ.get('IFA_MODE', 'text').lower()
  if mode == 'voice':
    try:
      voice = VoiceInput(tts)
      get_input = voice.get
      print('[voice mode] Say "hey jarvis" to start.')
    except Exception as exc:
      print(f'[warn] voice mode failed to init ({exc}); falling back to text.')
      get_input = default text input
  else:
    get_input = default text input
  ```
- Fallback notice is printed once, on startup; subsequent behavior is identical to text mode.

**Test scenarios:**
- Happy path (mocked): `IFA_MODE=voice`, all deps succeed → `VoiceInput.get()` returns transcribed text from mocked pipeline.
- Edge case: `IFA_MODE=voice`, openWakeWord download fails → orchestrator catches, prints notice, uses text input.
- Edge case: `IFA_MODE=text` (default) → voice path not exercised at all; existing text behavior preserved.
- Integration: orchestrator.run() with `IFA_MODE=voice` and mocked mic/wake/STT → accepts one "turn", calls agent_turn, exits on second turn returning "exit".

**Verification:** `IFA_MODE=voice python -m ifa.main` boots without crashing on a machine with working mic, openWakeWord models downloaded, and Ollama/qwen running. Typing still works with default env.

---

- [ ] **Unit 6: Launcher updates**

**Goal:** Run the assistant in voice mode from a single double-click.

**Requirements:** R8 (activation UX)

**Dependencies:** Unit 5 (voice mode functional)

**Files:**
- Create: `run-voice.bat` — sibling to `run.bat`; sets `IFA_MODE=voice` then calls `run.bat`.
- Create: `run-voice.ps1` + `run-voice.bat` wrapper pair (mirroring `run.ps1` + `run-ps.bat`) that set `$env:IFA_MODE='voice'` before delegating to the existing PowerShell launcher.
- Create: `run-voice.command` — sibling to `run.command` for macOS; exports `IFA_MODE=voice` before calling `run.command`.
- Modify: `README.md` — document voice-mode launch under the existing "Running Ifa" section.
- Leave the existing `run.bat`, `run.ps1`, `run-ps.bat`, and `run.command` untouched so text-mode users keep their current UX.

**Approach:**
- Simplest: add three sibling files (`run-voice.bat`, `run-voice.ps1`, `run-voice.command`) that each set `IFA_MODE=voice` then call the existing run.{bat|ps1|command}. Keeps the existing launchers untouched for text-mode users.
- Alternative: single launcher that prompts "[1] text / [2] voice" — more UX friction but one file fewer.
- Decision: siblings. Simpler code, simpler explanation.

**Test scenarios:**
- Manual: double-click `run-voice.bat` on Windows → Ifa boots with "[voice mode] Say 'hey jarvis' to start" visible.
- Manual: `IFA_MODE=voice` env-var present when Ifa subprocess starts (verify via printed startup log).

**Verification:** Launcher scripts run without syntax errors on Windows + macOS.

---

- [ ] **Unit 7: Stage 1 Unit 6b cleanup (gated on acceptance bench)**

**Goal:** Execute the deletion gate — run acceptance bench; if passes, delete dead code paths.

**Requirements:** R10, R11

**Dependencies:** Stage 1 Unit 6a (shipped), real Ollama running with qwen2.5:7b-instruct pulled

**Execution note:** Gate-first. Bench must exit 0 before any deletion.

**Files:**
- Run: `PYTHONPATH=. venv/bin/python -m ifa.tests.test_tool_call_bench` → verify exit code 0 + per-tool thresholds met.
- Delete: `ifa/core/brain.py`
- Delete: `ifa/skills/manager.py`
- Modify: `ifa/skills/reminder.py` — delete `handle()` method (`schedule()` stays as the tool-callable entry)
- Delete: `ifa/tests/test_pipeline.py` (0-byte stub)
- Modify: `ifa/tests/test_tts_service.py` — delete `ManagerDecouplingTests` class; replace with an AST guard confirming `orchestrator.py` imports `agent_turn` from `ifa.core.agent`.
- (Do **not** delete `ifa/tests/test_stt.py` — Unit 4 rewrites it into proper tests; keep the rewritten version.)

**Approach:**
- Run bench. If it fails: iterate on system prompt in `ifa/core/agent.py` (up to 3 rounds). If still failing after 3 rounds, pause and reassess per Stage 1 plan (qwen2.5:14b, Q5_K_M, hybrid rules+LLM).
- If it passes: execute deletions as a single commit. Regression-guard tests from Stage 1's `test_orchestrator_stage1.py` should stay green throughout.

**Test scenarios:**
- Pre-deletion: all 116 tests pass.
- Post-deletion: all tests still pass (minus `ManagerDecouplingTests` which is rewritten). `test_orchestrator_stage1.py` regression guards confirm no `detect_intent`/`extract_fact`/`extract_reminder`/`think`/`handle_with_intent` calls in non-test source.

**Verification:** `grep -rn '\bdetect_intent\|extract_fact\|extract_reminder\|\bthink\b\|handle_with_intent\b' ifa/ --include='*.py' | grep -v tests/` returns no results (symbols are gone from source).

---

## System-Wide Impact

- **Interaction graph:** `VoiceInput` replaces `input("You: ")` at the edge. `agent_turn` doesn't change. `TTSService` gains an `is_speaking` flag checked by the wake-word listener — this is a new cross-component dependency but it's one-directional (listener reads; TTS writes).
- **Thread topology:** Main thread runs the orchestrator loop (agent_turn → TTS). A dedicated background thread (started by `VoiceInput.__init__`) runs the wake-word + capture + STT pipeline and pushes completed utterances onto an internal `queue.Queue`. `VoiceInput.get()` blocks on `queue.get()`. This lets `tts.speak()` block the main thread for the full TTS duration without stalling the mic-facing work — and lets the wake-word thread poll `tts.is_speaking` from its own execution context. The `_speaking_lock` on TTSService is held only for nanosecond-scale critical sections, so contention is not a concern.
- **Error propagation:** Voice-pipeline init failures (missing models, bad mic, etc.) are caught at `VoiceInput.__init__` and fall back to text mode with a notice. Mid-session errors in the voice pipeline (e.g., mic unplugged) drop to exceptions — out of scope for v1.
- **State lifecycle risks:** `sounddevice.InputStream` is opened once and must be closed on orchestrator exit. If the process crashes, the OS reclaims it anyway (safe). The wake-word background thread is a daemon thread (`daemon=True`), so it exits when the main thread exits — no teardown hook required. With the Timer-free mute mechanism (Unit 1), there is no lingering Timer to cancel on shutdown.
- **API surface parity:** `TTSService.speak()` signature unchanged. `agent_turn` unchanged. Everything that currently works keeps working.
- **Integration coverage:** Three new tests that exercise the full voice chain with external deps mocked — prove the pieces wire together.
- **Observability (minimal):** Startup log line `[voice] mode=voice wake=hey_jarvis_v0.1 whisper=small.en threshold=0.5` on voice init, so a user reporting "it's not hearing me" can confirm which model/threshold is in play without code-reading. Per-turn: a single `[voice] wake detected (score=X.XX)` + `[voice] utterance captured (duration=Xs)` + `[voice] transcribed: "..."` log sequence at INFO level. Not sent to Ollama/TTS — local stderr only.
- **Unchanged invariants:** `IFA_MODE=text` is default; text mode behavior is byte-identical to post-Stage-1. SQLite schemas unchanged. Tool registry unchanged.

## Risks & Dependencies

| Risk | Mitigation |
|---|---|
| `openWakeWord` models fail to download (offline or firewall) | `VoiceInput.__init__` catches; falls back to text mode with actionable message. User can download models once with internet + then go offline. |
| faster-whisper internal `SileroVADModel` API breaks on upgrade | Pin `faster-whisper==1.2.1`; wrap VAD usage in try/except with energy-threshold fallback; add a test that constructs the VAD model to catch breakage early. |
| Wake-word false-positive rate is too high in user's environment | Start at 0.5 threshold; expose `IFA_WAKE_THRESHOLD` env var for tuning. If the false-positive rate is still unacceptable, Stage 2.5 adds push-to-talk fallback. |
| Mic self-trigger even with mute flag | The flag has a 500ms cooldown past end-of-TTS; if Ifa's echo reaches the mic with >500ms latency (unusual), we'd trigger. Tunable via `IFA_TTS_COOLDOWN_MS`. If this proves common, move to AEC (out of scope). |
| Transcription quality poor on noisy mics / distant speakers | `small.en` is the v1 default; users can upgrade to `medium.en` or `large-v3` via `IFA_WHISPER_MODEL`. No code changes needed. |
| openWakeWord "hey jarvis" is not "hey ifa" — identity feels off | Noted and accepted for v1; Stage 2.5 custom training resolves. |
| Unit 6b bench fails, blocking deletion | Iterate on system prompt up to 3 rounds; if still failing, pause Stage 2 deletion commit and ship voice loop on top of the dual-path state (dead code remains, but user-visible behavior is fine). |

## Documentation / Operational Notes

- Running Ifa in voice mode requires a working mic and speakers on the launch host — otherwise init fails and falls back to text.
- First-time voice launch downloads the openWakeWord `hey_jarvis_v0.1.onnx` model (~5MB) + faster-whisper `small.en` model (~470MB). Both cache locally; subsequent runs are fast.
- `IFA_MODE=voice` is the single switch. All other tunables are env vars with defaults (no config file needed).

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-23-conversation-and-tool-foundation-requirements.md](../brainstorms/2026-04-23-conversation-and-tool-foundation-requirements.md) (voice section R1-R6)
- **Stage 1 plan:** [docs/plans/2026-04-23-002-feat-stage1-llm-tool-dispatch-plan.md](2026-04-23-002-feat-stage1-llm-tool-dispatch-plan.md) (Stage 1 Unit 6b bundled here as Unit 7)
- **Prior shipped work:** commit `d7840fe` (cross-platform subprocess TTS — we extend with `is_speaking` flag)
- **openWakeWord docs:** `pip install openwakeword`, call `download_models(...)` once, instantiate with `inference_framework='onnx'`
- **faster-whisper bundled VAD:** `from faster_whisper.vad import SileroVADModel` — ONNX-only, no torch dep
