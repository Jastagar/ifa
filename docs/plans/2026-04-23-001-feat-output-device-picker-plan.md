---
title: feat: Cross-platform TTS backend (audibility fix)
type: feat
status: active
date: 2026-04-23
origin: docs/brainstorms/2026-04-23-output-device-picker-requirements.md
scope-note: Reduced from the original 6-unit picker plan after document-review consensus that the real pain (macOS silence) is solved by Unit 1 alone. Picker + persistence + mid-session triggers deferred until there is evidence of demand.
---

# feat: Cross-platform TTS backend (audibility fix)

## Overview

Replace Ifa's Windows-only PowerShell TTS backend with a cross-platform `pyttsx3` → WAV → `sounddevice.play()` pipeline on the system default output device. Unify the two separate `TTSService()` instantiations via explicit dependency injection (not a module-level singleton). That's the whole plan.

No picker, no device enumeration, no YAML config, no intent extension, no keyword dispatch, no threading.Event — all of that is deferred to follow-up work gated on actual evidence of demand. (Original requirements doc preserved at the origin link; the picker research is not lost, just paused.)

## Problem Frame

Ifa's TTS service at [ifa/services/tts_service.py](../../ifa/services/tts_service.py) shells out to Windows PowerShell's `System.Speech.Synthesizer` — which means **the user (on macOS) can't hear Ifa at all**. `TTSService` is also instantiated twice (once in [ifa/core/orchestrator.py:11](../../ifa/core/orchestrator.py) and once in [ifa/skills/manager.py:5](../../ifa/skills/manager.py)), so even fixing the backend in one place would leave reminder playback using the wrong instance.

Everything else the original brainstorm explored (device picking, persistence, mid-session triggers, fallback semantics) is predicated on user demand that hasn't been demonstrated. The OS default output is correct ~95% of the time on macOS and Windows; users can change the system default in one click if needed.

## Requirements Trace

From the origin brainstorm (narrow slice):
- **R13.** New cross-platform TTS backend. *This plan.*
- **R14.** All `tts.speak()` call sites route to the same backend instance. *This plan — via DI.*
- **R15.** Works on macOS and Windows; does not actively block Linux. *This plan.*

Explicitly NOT in this plan (deferred to follow-up):
- R1–R12 (picker UI, persistence, startup sequencing, mid-session triggers, fallback-on-missing-device).

## Scope Boundaries

- **Not in scope:** device selection, device picker, `ifa/config/settings.yaml`, `ifa/services/device_store.py`, `ifa/ui/picker.py`, `ifa/skills/audio_device.py`, `detect_intent` audio label, keyword command dispatch, startup picker sequencing, `threading.Event` for picker/reminder coordination, Linux smoke test.
- **Not in scope:** any device other than the OS default. TTS always plays on the system default output after this plan. If the user wants AirPods, they switch AirPods as the system default — the OS already has UI for that.
- **Not in scope:** input device selection, volume, voice quality tuning.

### Deferred to Separate Tasks

- **Output device picker and everything downstream of it (original R1–R12).** Deferred pending evidence that the user actually wants to route to a non-default device from inside Ifa rather than using the OS's built-in default-switcher. Requirements doc at [docs/brainstorms/2026-04-23-output-device-picker-requirements.md](../brainstorms/2026-04-23-output-device-picker-requirements.md) retains the research for whenever that evidence arrives.

## Context & Research

### Relevant Code and Patterns

- [ifa/services/tts_service.py](../../ifa/services/tts_service.py) — 23-line PowerShell backend; replace wholesale.
- [ifa/voice/tts_engines/pyttsx_engine.py](../../ifa/voice/tts_engines/pyttsx_engine.py) — unused `pyttsx3` wrapper using `engine.say()`. Delete; no callers (grep-confirmed).
- [ifa/core/orchestrator.py:11](../../ifa/core/orchestrator.py) — module-level `tts = TTSService()`. Stays, but becomes the *only* construction site.
- [ifa/skills/manager.py:5](../../ifa/skills/manager.py) — second `tts = TTSService()`. **Delete this line** and receive `tts` via the existing DI pattern already used for `ReminderSkill(tts)` on line 9.
- [ifa/tests/test_pipeline.py](../../ifa/tests/test_pipeline.py) — test conventions.

### External References

- `pyttsx3.save_to_file(text, path)` + `runAndWait()` writes a WAV on both macOS (NSSpeechSynthesizer/AVSpeechSynthesizer) and Windows (SAPI).
- `sounddevice.play(data, samplerate)` takes a **numpy array** — WAV files must be decoded first. Options: `soundfile.read()` (thin libsndfile wrapper; add to `requirements.txt`) or stdlib `wave` + `numpy.frombuffer`.
- **Thread-safety caveat:** `pyttsx3` on macOS delegates to Cocoa's `NSRunLoop` which is not safe to drive from background threads. `resume_reminders()` in orchestrator.py spawns daemon threads that call `tts.speak()`. This is a real constraint that shapes the design below.

## Key Technical Decisions

- **Backend = `pyttsx3.save_to_file()` → decode → `sounddevice.play(device=None)`.** Offline, cross-platform, targets the OS default output. No device parameter — by design.
- **DI over singleton.** Construct `TTSService()` exactly once, at the top of `ifa/core/orchestrator.py`. Pass it into `handle_with_intent()` as a new parameter, which forwards to skills that need it (currently only `ReminderSkill`). This closes the duplicate-instance problem using the codebase's *existing* pattern (`ReminderSkill(tts)` on [ifa/skills/manager.py:9](../../ifa/skills/manager.py)) — no new file, no module-level `get_tts()` accessor, no lazy-init ordering trap.
- **Thread-safety strategy = main-thread synthesis queue.** `TTSService.speak()` from the main thread runs synthesis directly. `TTSService.speak()` from any other thread (i.e., reminder daemon threads) posts a request to a `queue.Queue` drained on each main-loop iteration in `run()`. This sidesteps the macOS Cocoa-from-daemon-thread hazard without introducing platform-specific code paths. The reminder threads wait on the queue-drain result to preserve "speak() blocks until done" semantics.
- **WAV decode = `soundfile.read()` (new dep).** Adds `soundfile` to `ifa/requirements.txt`. Chosen over stdlib `wave` + `numpy.frombuffer` because `pyttsx3` on macOS vs Windows may produce different WAV subtypes / bit depths; `soundfile` normalizes.
- **PortAudioError handling = catch, notice, no-op.** If `sd.play()` or `sd.wait()` raises `PortAudioError` (very rare on system default), print a one-line notice via `rich.console.Console` and return normally. No device fallback (there's no picker yet), no retry counter.

## Open Questions

### Resolved During Planning

- **Which backend?** → `pyttsx3.save_to_file()` + `soundfile.read()` + `sounddevice.play(device=None)`.
- **Singleton vs DI for unifying the two TTSService instances?** → DI. Use the existing `ReminderSkill(tts)` pattern.
- **Thread-safety strategy?** → Main-thread synthesis queue.
- **Add soundfile dependency?** → Yes.

### Deferred to Implementation

- **Exact queue-drain cadence for reminder-thread speak requests.** The main loop blocks on `input("You: ")`. Draining the queue only between user inputs is acceptable but means a reminder waits for the user to hit enter. Alternative: spawn a dedicated main-thread consumer. Decide when implementing.
- **pyttsx3 voice selection.** Plan ships with default voice on each platform. If quality is poor, a platform-specific voice id can be set; a follow-up concern.

## Implementation Units

- [ ] **Unit 1: Cross-platform TTS backend with main-thread synthesis queue**

**Goal:** Replace PowerShell-based `TTSService` with a cross-platform pyttsx3 + soundfile + sounddevice pipeline on system default. Unify the two call sites via DI. Keep thread-safety invariants by routing daemon-thread speak requests through a main-thread queue.

**Requirements:** R13, R14, R15

**Dependencies:** None

**Files:**
- Modify: `ifa/services/tts_service.py` (replace PowerShell implementation; add queue-based thread forwarding)
- Modify: `ifa/skills/manager.py` (remove module-level `TTSService()`; accept `tts` as a parameter to `handle_with_intent`)
- Modify: `ifa/core/orchestrator.py` (pass `tts` into `handle_with_intent`; drain main-thread synthesis queue between user inputs)
- Modify: `ifa/requirements.txt` (add `soundfile`)
- Delete: `ifa/voice/tts_engines/pyttsx_engine.py` (unused; grep-confirmed no callers)
- Test: `ifa/tests/test_tts_service.py` (new)

**Approach:**
- `TTSService.__init__()` creates a `queue.Queue` for cross-thread speak requests and captures the main thread's ident via `threading.get_ident()`.
- `speak(text)`:
  - If called on the main thread: synthesize via `pyttsx3.save_to_file()` → decode via `soundfile.read()` → `sounddevice.play(data, sr)` → `sounddevice.wait()` → delete temp WAV in `finally`.
  - If called on another thread: wrap the text in a `(text, threading.Event, result_slot)` tuple, push to the queue, block on the event until the main-loop drain completes, then return.
  - Wrap both `sd.play()` and `sd.wait()` in `try/except sounddevice.PortAudioError` → print one-line notice via `rich.console.Console`, do not re-raise.
- `drain_queue()` (new method): called once per iteration of the orchestrator's main `while True:` loop. Pops all pending requests and synthesizes them in order; sets the event on each to unblock the caller.
- In `ifa/skills/manager.py`: delete the module-level `tts = TTSService()`. Change `skills = {...}` registration so that `ReminderSkill` is constructed lazily when needed, OR change `handle_with_intent(intent, text)` signature to `handle_with_intent(intent, text, tts)` and construct `ReminderSkill(tts)` at handle time. Prefer the latter — it's one-line change.
- In `ifa/core/orchestrator.py`: construct `tts = TTSService()` (stays where it is); pass it into `handle_with_intent(intent, user_input, tts)`; call `tts.drain_queue()` at the top of each `while True:` iteration before `get_input()`.

**Execution note:** Start with a failing integration test that reproduces "macOS user can hear Ifa speak" — importing the module and calling `speak("hello")` on main thread should actually produce audio on macOS. The PowerShell backend fails this test silently today; the new backend must pass it before any other change is accepted.

**Patterns to follow:**
- Existing DI pattern: [ifa/skills/manager.py:9](../../ifa/skills/manager.py) `ReminderSkill(tts)` — same shape extended to `handle_with_intent`.
- Threading discipline: [ifa/core/orchestrator.py:25-39](../../ifa/core/orchestrator.py) `resume_reminders` daemon-thread pattern — reminder threads call `tts.speak()` exactly as they do today; the thread-dispatch logic lives inside `TTSService`, not at the call sites.

**Test scenarios:**
- Happy path: `TTSService().speak("hello")` on main thread produces audio on system default (assert `sd.play()` was called with `device=None` and a non-zero numpy array)
- Happy path (cross-thread): `threading.Thread(target=lambda: tts.speak("hi")).start()` posts to queue; main-thread `drain_queue()` picks it up; the thread unblocks after synthesis completes
- Edge case: `speak("")` on main thread is a no-op (no file written, no playback attempt)
- Error path: `sd.play()` raises `PortAudioError` → speak() prints notice via `rich.console.Console`, does NOT raise
- Error path: `sd.wait()` raises `PortAudioError` → same behavior
- Integration: `handle_with_intent("reminder", "remind me in 1 second", tts)` creates a reminder; after 1s the reminder daemon thread calls `speak()`; the next `drain_queue()` call in the main loop produces audible output
- Integration: `ifa.skills.manager` no longer constructs its own `TTSService()` — grep asserts the module only has one class-level `TTSService` reference (in a type hint or similar)
- Cross-platform: the test suite runs on macOS and Windows CI (if available) — `speak("hi")` produces a WAV file, decoded, played without error on both

**Verification:**
- `python -m ifa.main` on macOS produces audible TTS — the user can actually hear Ifa for the first time
- A reminder fired from a daemon thread produces audible output on the next main-loop iteration, with no crash
- `grep -n 'TTSService()' ifa/` returns exactly one construction site (in `ifa/core/orchestrator.py`)

## System-Wide Impact

- **Interaction graph:** `TTSService` is now constructed once in `ifa/core/orchestrator.py` and threaded through `handle_with_intent` to any skill that needs it. Any future skill that wants to speak must declare `tts` in its constructor or receive it as a handle-time parameter.
- **Error propagation:** `sounddevice.PortAudioError` is caught inside `TTSService.speak()` and never reaches callers. Callers see `speak()` as infallible-from-outside.
- **API surface parity:** `speak()` signature is unchanged. Existing callers in `ifa/core/orchestrator.py` and `ifa/skills/reminder.py` need no changes beyond what's described in Files.
- **Unchanged invariants:** `detect_intent`'s return type/values (still `time|reminder|none`). `handle_with_intent`'s return type (`Optional[str]`). SQLite reminders schema. The orchestrator's input loop shape (`input("You: ")` → intent → skill → think → speak).
- **Import-order invariant:** `TTSService` is not constructed at import time anywhere. Constructing it in `run()` before the main loop starts means `pyttsx3.init()` (which can stall briefly on first-use driver probing) happens inside `run()`, not on module load.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `pyttsx3.save_to_file()` from daemon thread silently fails on macOS | Main-thread synthesis queue routes all cross-thread speak requests through the main loop. Daemon threads never touch pyttsx3 directly. |
| `pyttsx3` default voice on macOS is subjectively poor | Acceptable for v1. Follow-up may set a specific voice id via `engine.setProperty('voice', ...)`. |
| Queue drain only between user inputs means a reminder waits for `input()` to return | Documented. Acceptable given current reminder cadence. A dedicated main-thread consumer is a follow-up if needed. |
| `soundfile` adds a new dependency (libsndfile C lib) | `soundfile` wheels include the shared library on macOS/Windows; no manual install. Stdlib `wave` fallback is possible if the dep proves painful — noted in Deferred to Implementation. |
| `ifa/core/orchestrator.py` change ripples into `handle_with_intent`'s signature | Only one existing call site. The change is mechanical. |

## Documentation / Operational Notes

- If the project README mentions TTS, update it to note cross-platform support (currently README is minimal — check before editing).
- `soundfile` is added to `ifa/requirements.txt`. No runtime install surprises on macOS/Windows (wheels bundle libsndfile); Linux users would need `libsndfile1` installed, which is common.

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-23-output-device-picker-requirements.md](../brainstorms/2026-04-23-output-device-picker-requirements.md)
- **Scope reduction rationale:** Document-review on the original 6-unit plan surfaced P0 technical blockers (pyttsx3 daemon-thread hazard, WAV decode gap) and P1 product concerns (no evidence for device-switching demand, maintenance surface disproportionate to a solo-dev personal tool). Consensus across product-lens, scope-guardian, and adversarial reviewers: ship Unit 1 alone as v1; revisit the picker when there's evidence it's wanted.
- **Review findings this plan closes:** O1 (duplicate TTSService — via DI, not singleton), E1 (no keyword dispatch to break — deferred), P0 WAV decode (via soundfile), P0 pyttsx3 thread-safety (via main-thread queue), P1 maintenance surface (7 new modules → 0 new modules, 1 new dep).
- Related code: [ifa/services/tts_service.py](../../ifa/services/tts_service.py), [ifa/core/orchestrator.py](../../ifa/core/orchestrator.py), [ifa/skills/manager.py](../../ifa/skills/manager.py)
