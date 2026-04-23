---
date: 2026-04-23
topic: output-device-picker
---

# Output Device Picker (TUI)

## Problem Frame

Ifa currently plays TTS through whatever the OS picks — there's no way for the user to choose which speaker, headphones, or other output device the assistant talks through. On top of that, the current TTS backend (`ifa/services/tts_service.py`) shells out to Windows PowerShell's `System.Speech.Synthesizer`, which is Windows-only and exposes no device-selection API at all. The user (on macOS) can't hear Ifa today, and even on Windows they can't route voice to a non-default device.

This brainstorm defines a terminal UI ("picker") for picking the output device, with the choice persisted across runs and changeable mid-session.

## Requirements

**Picker UI**
- R1. A terminal-based picker lists all available audio output devices by human-readable name.
- R2. The current/saved device is clearly indicated (e.g. arrow, highlight, or `*` marker).
- R3. The user can select a device with arrow keys + enter, or by typing its number.
- R4. Cancelling the picker (Esc / Ctrl-C) leaves the current selection unchanged — it does not abort Ifa.

**Trigger behavior**
- R5. On startup, if no saved device exists, the picker opens automatically before the main loop starts.
- R6. On startup, if a saved device exists and is available, Ifa boots straight into the main loop without prompting.
- R7. Mid-session, the picker is reachable two ways: (a) a reserved keyword command typed at the `You:` prompt (e.g. `/audio`, `/output`), and (b) a natural-language intent (e.g. "change output device", "switch my speaker") routed through the existing `detect_intent` / `handle_with_intent` pipeline.
- R8. Reserved keyword commands bypass the LLM and intent classifier entirely.

**Persistence and fallback**
- R9. The selected device is persisted so it survives Ifa restarts.
- R10. If the saved device is not present at startup (e.g. AirPods disconnected), Ifa falls back to the system default, prints a visible notice, and speaks a short notification ("AirPods not found, using system default").
- R11. Fallback does not block startup — the main loop begins immediately on the default device.
- R12. The user can re-open the picker at any time via the mid-session triggers in R7 to recover from fallback.

**TTS routing (bundled scope)**
- R13. The TTS backend is replaced with one that supports targeting a specific output device. The PowerShell backend is removed or guarded so it can never run on non-Windows platforms.
- R14. Once a device is selected, all subsequent `tts.speak()` calls in the orchestrator (`ifa/core/orchestrator.py`) route audio to that device — including reminder playback (`resume_reminders`) fired from background threads.
- R15. TTS works on both macOS and Windows with the same selection flow. (Linux is not a current target but the design should not actively block it.)

## Success Criteria

- A first-time user on macOS or Windows starts Ifa, sees the picker, chooses a device, and hears Ifa speak through that device.
- Restarting Ifa with the same device plugged in boots silently to the saved device — no picker.
- Unplugging the saved device before restart produces a visible + spoken fallback notice on the system default, and Ifa is still usable.
- Typing `/audio` (or equivalent) mid-session, or saying "change output device," reliably reopens the picker.
- Removing the PowerShell TTS call path closes a current bug: macOS users can actually hear Ifa.

## Scope Boundaries

- **Input device selection is not in scope.** This picker is output-only; microphone selection stays on the OS default. (Worth revisiting once STT mode is wired up.)
- **Per-skill device routing is not in scope.** Reminders, responses, and every other speak() call share the same output device.
- **Volume control is not in scope.** Only device selection.
- **No GUI / system tray / menu bar.** Terminal TUI only. Desktop UI was considered and explicitly deferred.
- **No hot-swap / live re-route of in-flight utterances.** Changing devices mid-utterance may cut the current line; that's acceptable.
- **No multi-device simultaneous playback.** Exactly one output device at a time.

## Key Decisions

- **TUI over desktop UI**: fits the existing terminal-only experience; avoids adding a GUI framework and a background-process lifecycle.
- **Startup + on-demand trigger model**: prompt only when necessary (first run / missing device), but always reachable via keyword + intent.
- **Both keyword and intent triggers**: keyword is the reliable/deterministic path; intent is the natural voice-first path. Keyword bypasses intent classification so recovery always works even if the LLM misclassifies.
- **Notify-on-fallback over force-picker-on-fallback**: keeps startup fast and predictable; user can still reach the picker immediately if they want.
- **Bundle TTS backend swap into this feature**: a device picker is meaningless without a backend that accepts a device. The current PowerShell backend is also already broken for this user's platform.

## Dependencies / Assumptions

- `sounddevice`, `rich`, `typer`, `edge-tts`, `pyttsx3`, and `PyYAML` are already in `ifa/requirements.txt`. No new runtime dependency is expected — though the final TTS backend choice is deferred to planning.
- Device names (not indices) are assumed stable enough across reboots to be used as the persistence key. Indices are known to shift when devices are plugged/unplugged and should not be the persistence key.
- Reminders (`resume_reminders` in `ifa/core/orchestrator.py`) spawn background threads that call `tts.speak()`; the chosen backend must be safe to call from threads, or be serialized behind a lock.

## Outstanding Questions

### Resolve Before Planning

_(none — all product decisions are resolved.)_

### Deferred to Planning

- [Affects R13, R14][Technical] Which TTS backend? Candidates: (a) `edge-tts` synth → WAV bytes → `sounddevice` playback to chosen device, (b) `pyttsx3` `save_to_file` → `sounddevice` playback, (c) something else. Trade-offs: quality, offline support, latency, cross-platform behavior.
- [Affects R9][Technical] Where to persist the selected device? Candidates: a new `ifa/config/settings.yaml` (PyYAML is already a dep and the `config/` module is empty), the existing SQLite DB used for reminders, or a user-dir file like `~/.ifa/config.yaml`. Pick one consistent with future config needs.
- [Affects R10][Needs research] How does `sounddevice` behave on each platform when a named device disappears between process starts — does the name still resolve, does it raise, or does it silently route to default? Confirm before writing fallback logic.
- [Affects R7, R8][Technical] Exact keyword surface. Candidates: `/audio`, `/output`, `/device`, plain-text `change device`. Planning should pick one primary plus a short alias list.
- [Affects R14][Technical] Thread-safety of whichever playback path is chosen, given `resume_reminders` calls `tts.speak()` from daemon threads.

## Next Steps

-> `/ce:plan` for structured implementation planning
