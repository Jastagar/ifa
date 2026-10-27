# Ifa: architecture

Ifa is a personal voice assistant that runs on your own PC: a local LLM in Ollama, a custom wake word, Hinglish speech-to-text, a cloned Chatterbox voice, and tools for apps, music, reminders, memory, notifications and terminal commands. An Android app acts as a remote mic, speaker and notification source.

This page is the map. Each area has its own document with the decisions and the reasons behind them:

| Area | Document |
|---|---|
| Agent loop, tools, guards, memory, persona | [design/agent-loop.md](design/agent-loop.md) |
| Wake word → STT → TTS, streaming, fillers, latency | [design/speech-pipeline.md](design/speech-pipeline.md) |
| Phone notifications: inbox, urgency, summaries, Apps tab | [design/notifications.md](design/notifications.md) |
| Android app and its WebSocket protocol | [design/phone-app.md](design/phone-app.md) |
| Only your voice (speaker verification) | [design/speaker-verification.md](design/speaker-verification.md) |
| Spotify, apps, websites, run_command, WhatsApp | [design/tools.md](design/tools.md) |
| Launcher, setup, package versions, offline mode | [design/launcher-and-setup.md](design/launcher-and-setup.md) |
| **Every decision, with why and what was rejected** | [decisions.md](decisions.md) |
| **What's next, parked, unverified or limited** | [roadmap.md](roadmap.md) |

---

## The system

```
                    ┌──────────── PC ─────────────────────────────────────────────────────────┐
 Android app        │                                                                         │
 ─────────────      │  launcher (localhost page)  ── starts/stops ──►  python -m ifa.main     │
 wake word "Ifa" ───┼── WebSocket :8789 ──► phone_server ──┐            (supervisor restarts  │
 mic / speaker      │   (UDP :8788 discovery)              │             on exit code 75)     │
 notifications ─────┼──────────────────────────────────────┤                                  │
 Apps tab, settings │                                      ▼                                  │
                    │  PC mic ─► wake word ─► VAD ─► voice check ─► Whisper ─┐                │
                    │                                                        ▼                │
                    │            NotificationCenter ◄──── agent turn (ReAct, ≤ 3 tool rounds) │
                    │                                       │      ▲                          │
                    │                                       ▼      │                          │
                    │                       tools (apps, Spotify, run_command, memory, …)     │
                    │                                       │                                 │
                    │       Ollama (one model) ◄────────────┘                                 │
                    │                                                                         │
                    │  reply sentences ─► Chatterbox ─► PC speakers, or streamed to the phone │
                    └─────────────────────────────────────────────────────────────────────────┘
```

**Principles that run through all of it**
- **Local first.** Ifa needs no internet once set up ([offline mode](design/launcher-and-setup.md#offline-mode-2026-10-21)). Only the online skills (Spotify, web, WhatsApp, n8n) need it.
- **One model, one memory, one queue.** PC and phone turns share memory and never overlap.
- **Steer with the prompt, enforce with code.** A 9B model fails in predictable ways. Fabricated tool results, claimed actions, command confirmation and the voice check are all enforced in code.
- **Third-party text is data.** Notifications, web pages and command output reach the model only inside per-turn nonce markers or `<notification>` blocks. Your own settings (app notes) are passed as trusted instructions, outside those blocks.
- **Measure, then change.** Time to first speech, tool-choice probes and model comparisons justified each change. The numbers are in the design documents.

## One voice turn

1. **Wake word:** openWakeWord `ifa.onnx`, on the PC mic or on the phone.
2. **Capture:** Silero VAD ends the utterance on silence. The phone sends its own pause and threshold settings.
3. **Voice check:** under about 1 s of speech is trusted only right after a matched utterance; otherwise a CAM++ embedding is scored against your voiceprint. Not you means ignored, never transcribed.
4. **Whisper:** Oriserve Hinglish Apex, `en` token.
5. **Goodbyes** ("that's all") end the turn with no model call.
6. **Agent turn:** system prompt and persona, last 4 whole turns, and an unread-notification hint. Ollama streams, tools run, results come back marked as data.
7. **Speech:**
   - a filler clip if nothing is spoken within 650 ms (or right away, when a tool is called)
   - sentences go to Chatterbox as they arrive
   - the audio plays on the PC, or streams to the phone
8. **Follow-up:** listen briefly without the wake word.

## Where things live

| Path | What it is |
|---|---|
| [ifa/main.py](../ifa/main.py), [ifa/supervisor.py](../ifa/supervisor.py) | Entry point: load `.env`, pick offline/online, run under the restart supervisor |
| [ifa/core/orchestrator.py](../ifa/core/orchestrator.py) | Startup and the main loop; `run_turn` (fillers, ordering) |
| [ifa/core/agent_stream.py](../ifa/core/agent_stream.py) | The ReAct loop, streaming, guards |
| [ifa/core/memory.py](../ifa/core/memory.py), [ifa/services/memory_store.py](../ifa/services/memory_store.py) | Short-term turns; long-term memories by topic (SQLite) |
| [ifa/core/dismiss.py](../ifa/core/dismiss.py), [ifa/core/personality.py](../ifa/core/personality.py), [ifa/services/prompts.py](../ifa/services/prompts.py) | Goodbye detection; tool-data framing; editable prompts (persona, urgency, summary) |
| [ifa/tools/](../ifa/tools/) | One file per tool; `registry.py` validates, dispatches, marks results as data |
| [ifa/skills/](../ifa/skills/) | The code tools call: Spotify, apps, browser, commands, reminders, vision, games |
| [ifa/voice/](../ifa/voice/) | Wake word, capture, STT, cues, speaker verification, enrollment |
| [ifa/services/](../ifa/services/) | TTS, fillers, notifications, phone server, pairing, Ollama client, custom apps, DB |
| [ifa/utils/](../ifa/utils/) | Latency timer, device choice, offline mode, speech queue |
| [launcher/](../launcher/) | The local setup and settings page |
| [scripts/](../scripts/) | `install_packages`, `setup_voice_models`, `notify-test`, `smoke_voice` |
| [ifa/requirements.txt](../ifa/requirements.txt), [ifa/constraints.txt](../ifa/constraints.txt) | What to install; the exact tested versions |
| `~/Projects/ifa.app` | The Android app (separate repo) |

Runtime files, not committed: `.env`, `apps.json`, `prompts.json`, `voiceprint.json`, `.phone_token`, `.spotify_token`, `ifa.db`, `models/`.

## Tests

```
PYTHONPATH=. venv/bin/python -m unittest discover ifa/tests
```

- The suite mocks the LLM, audio devices and heavy models (a scripted fake LLM drives the agent-loop tests).
- `test_tool_call_bench.py` is a manual bench against a live Ollama, with sandboxed tool handlers.
- `test_speaker.py` includes a real-model test when `IFA_SR_DATA` points at sherpa-onnx's sample recordings.
- On an Apple Silicon Mac, Whisper and Chatterbox run on the Apple GPU (`mps`); sentence splitting is plain Python ([ifa/utils/sentences.py](../ifa/utils/sentences.py)).

## When something's wrong

| Symptom | Look at |
|---|---|
| Install fails or torch/transformers errors | Launcher → Install packages (shows wrong versions); [launcher-and-setup.md](design/launcher-and-setup.md#package-versions-2026-10-21) |
| Ifa needs internet at startup | The `[env] offline/online` line; Setup → Download voice models |
| Ifa ignores you | `[voice] ignored: … (score …)` lines; `IFA_SPEAKER_THRESHOLD`; enroll again |
| Notifications never announce | The app's Apps tab switch; `[notify] … urgent=… (reason)` in the console |
| A tool isn't called / wrong tool | The tool's description in `ifa/tools/*.py`; [agent-loop.md](design/agent-loop.md#tool-selection-descriptions-matter) |
| "Playing X" but nothing plays | `[agent] reply claimed an action without a tool call`; Spotify login / device |
| Slow first word | The per-turn latency report; `IFA_LATENCY_LOG` |

## History

| When | What | Where |
|---|---|---|
| April 2026 | Stage 1: LLM tool dispatch. Stage 2: voice input. Stage 3: custom wake word | [plans/](plans/), [brainstorms/](brainstorms/) |
| 2026-10-15 → 10-21 | Dead-code removal, ReAct loop, streaming and fillers, notifications, launcher, tools, phone app, speaker verification, offline mode, package lock | [decisions.md](decisions.md) |
