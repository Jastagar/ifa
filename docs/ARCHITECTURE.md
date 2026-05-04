# Ifa — architectural walkthrough

A learning-oriented tour of the codebase. Read this once, then dive into the files in the order suggested.

---

## What Ifa is

A personal AI assistant that runs entirely on your own machine. You talk to it (text or voice); it understands the request, picks tools (get the time, set a reminder, remember a fact, trigger an n8n workflow, or just chat), executes them, and replies. The LLM lives in [Ollama](https://ollama.com/) (`qwen2.5:7b-instruct`); the voice stack uses [openWakeWord](https://github.com/dscripka/openWakeWord) for wake-word, [Silero VAD](https://github.com/snakers4/silero-vad) for end-of-turn, and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription; the TTS uses the OS native engines (`say`/SAPI/`espeak`).

It is intentionally **single-user, single-machine, offline-first**. Models are pre-downloaded once; runtime sets `HF_HUB_OFFLINE=1` so it never phones home.

---

## How a turn flows

### Text mode (default)

```
                 ┌────────────────────────────────────────────────┐
                 │                                                │
 user types ─────►  orchestrator.run()  ◄──── tts.speak(reply) ──┘
                       │
                       ▼
                 input_mode.get()                ifa/voice/input.py (_TextMode)
                       │  returns "what time is it"
                       ▼
                 agent_turn(text, ctx, memory)   ifa/core/agent.py
                       │   1. build messages with system prompt
                       │      + recent memory + new user text
                       │   2. POST to Ollama /api/chat with tools=[...]
                       │   3. LLM emits {tool_calls: [{name: "get_time", ...}]}
                       │   4. registry.dispatch("get_time", args, ctx)
                       │   5. wrap result with delimit_as_data(nonce, ...)
                       │   6. send result back to LLM
                       │   7. LLM emits final assistant text "It's 3:45 PM"
                       ▼
                 reply (string)
                       │
                       ▼
                 tts.speak(reply)                ifa/services/tts_service.py
                       │  (subprocess: say/SAPI/espeak)
                       ▼
                 input_mode.arm_followup()       no-op in text mode
```

### Voice mode (`IFA_MODE=voice`)

```
       MAIN THREAD                              VOICE THREAD (daemon)

       orchestrator.run()
            │                                   ifa/voice/input.py — VoiceInput
            ▼                                   ┌──────────────────────────────┐
       input_mode.get()  ◄────── queue.put ─────│ wait_for_wake (1280-sample)  │
            │   (blocks)                         │  ifa/voice/wake_word.py     │
            ▼                                   │ capture_utterance (512)      │
       agent_turn(text, ...)                    │  ifa/voice/capture.py        │
            │                                   │ transcribe_array(audio)      │
            ▼                                   │  ifa/voice/stt.py            │
       tts.speak(reply)                         │                              │
            │   (is_speaking=True               └──────────────────────────────┘
            │    during playback +                    ▲
            │    cooldown)                            │ shared sounddevice.InputStream
            ▼                                         │ at 16 kHz mono float32
       input_mode.arm_followup()
            │   (sets _turn_complete event)
            ▼
       (loop)
```

The voice thread feeds the same `agent_turn` the text-mode loop does. The main thread can block on `tts.speak()` for the full TTS duration without stalling the mic — which would otherwise mean missing the user's follow-up.

---

## Where everything lives (call-graph order)

### Entry point

| File | Role |
|---|---|
| **[ifa/main.py](../ifa/main.py)** | Loads `.env`, prints config summary, calls `orchestrator.run()`. This is the file Python invokes. |

### Orchestration

| File | Role |
|---|---|
| **[ifa/core/orchestrator.py](../ifa/core/orchestrator.py)** | The main loop. Does startup (health check Ollama, load n8n config, init DB, build TTS + AgentContext, register tools, resume pending reminders), then loops on `input_mode.get()` → `agent_turn` → `tts.speak` → `arm_followup`. |
| **[ifa/core/agent.py](../ifa/core/agent.py)** | `agent_turn(user_text, ctx, memory)` — one conversational turn. Builds messages with system prompt + memory + new input, POSTs to Ollama with the tool schema, dispatches any tool calls the LLM emits, and feeds the results back. Stage 1's `MAX_ITERATIONS=1` caps tool-call hops per turn. |
| **[ifa/core/context.py](../ifa/core/context.py)** | `AgentContext` — a frozen-ish dataclass carrying `tts`, `db_path`, and `n8n_config`. Threaded through every tool handler so handlers can side-effect without module-level singletons. |
| **[ifa/core/memory.py](../ifa/core/memory.py)** | `Memory` — bounded list of `{role, content}` dicts. Short-term chat history, NOT long-term knowledge. Long-term facts live in SQLite via the `remember_fact` tool. |

### Tools (LLM dispatch surface)

| File | Role |
|---|---|
| **[ifa/tools/registry.py](../ifa/tools/registry.py)** | The `Tool` dataclass + `register/dispatch/as_ollama_schema` helpers. `dispatch` JSON-Schema-validates args before invoking the handler; `delimit_as_data` wraps results with a per-turn nonce so a crafted tool result can't inject prompt instructions. |
| **[ifa/tools/__init__.py](../ifa/tools/__init__.py)** | `register_all()` — explicit registration of every Stage 1 tool. Called once at startup. |
| **[ifa/tools/time.py](../ifa/tools/time.py)** | `get_time` — read this if you want a minimal example of declaring a tool. |
| **[ifa/tools/reminder.py](../ifa/tools/reminder.py)** | `set_reminder` — adapter that calls `ReminderSkill.schedule(task, seconds)`. |
| **[ifa/tools/memory.py](../ifa/tools/memory.py)** | `remember_fact` — INSERTs into the `facts` table, with retrieval via `load_facts()` (called by `agent.py` to inject facts into the system prompt). |
| **[ifa/tools/n8n.py](../ifa/tools/n8n.py)** | `call_n8n_workflow` — POSTs to a configured webhook. Per-workflow JSON Schemas in `ifa/config/n8n_workflows.yaml`. |

### Services (stateful infrastructure)

| File | Role |
|---|---|
| **[ifa/services/tts_service.py](../ifa/services/tts_service.py)** | Cross-platform TTS via subprocess (`say -o`+`afplay` / PowerShell SAPI / `espeak`). Exposes `is_speaking` so the wake-word listener can mute its mic during Ifa's reply (and for 500 ms after — the cooldown window). |
| **[ifa/services/db.py](../ifa/services/db.py)** | SQLite schema + `init_db`. WAL mode. Two tables: `reminders` and `facts`. |
| **[ifa/services/ollama_client.py](../ifa/services/ollama_client.py)** | HTTP wrapper around Ollama's `/api/chat` and `/api/tags`. `build_tool_result_message` isolates the `tool_name` field quirk so callers don't drift. |

### Voice pipeline (only loaded in `IFA_MODE=voice`)

| File | Role |
|---|---|
| **[ifa/voice/input.py](../ifa/voice/input.py)** | `init_input(tts)` returns the input mode (text or voice). `VoiceInput` opens the mic stream once and runs wake→capture→transcribe on a daemon thread feeding a `queue.Queue`. Read the `_run_loop` method's docstring to understand the mute / drain / arm_followup interactions. |
| **[ifa/voice/wake_word.py](../ifa/voice/wake_word.py)** | `WakeWordListener` — openWakeWord wrapper. Default model is the bundled `ifa/models/ifa.onnx` (custom-trained); `IFA_WAKE_MODEL` env var overrides with built-in name or path. Includes the missing-file fallback (logs WARNING, falls back to `hey_mycroft`). |
| **[ifa/voice/capture.py](../ifa/voice/capture.py)** | `capture_utterance(read_chunk)` — 512-sample chunks fed to Silero VAD. Cuts on 1.5 s trailing silence or 30 s max. Has a `start_timeout_ms` for the follow-up window. |
| **[ifa/voice/stt.py](../ifa/voice/stt.py)** | `transcribe_array(audio)` — faster-whisper on a numpy array. Lazy model load. `IFA_WHISPER_DEVICE=auto` tries CUDA first, falls back to CPU. |

### Skills (mostly historical)

| File | Role |
|---|---|
| **[ifa/skills/base.py](../ifa/skills/base.py)** | Pre-Stage-1 base class. Read its docstring for the "don't subclass this in new code" guidance. |
| **[ifa/skills/reminder.py](../ifa/skills/reminder.py)** | `ReminderSkill.schedule(task, seconds)` — the only live method. Called by the `set_reminder` tool adapter. |

### Launchers (operator UX)

| File | Role |
|---|---|
| **[run.bat](../run.bat) / [run.ps1](../run.ps1) / [run.command](../run.command)** | Self-healing launchers: create venv, install deps, start Ollama, pull qwen if needed, pre-download voice models, then `python -m ifa.main` with `HF_HUB_OFFLINE=1`. |
| **[run-voice.bat](../run-voice.bat) / [run-voice-ps.bat](../run-voice-ps.bat) / [run-voice.command](../run-voice.command)** | Thin wrappers: set `IFA_MODE=voice`, then delegate to the text-mode launcher. |
| **[scripts/setup_voice_models.py](../scripts/setup_voice_models.py)** | Pre-downloads openWakeWord + faster-whisper models. Idempotent. Called by every launcher before the runtime launch. |
| **[scripts/smoke_voice.py](../scripts/smoke_voice.py)** | Manual smoke modes: `mic`, `record`, `capture`, `transcribe`, `diagnose`. Use these when voice mode misbehaves to triangulate which stage is at fault. |

### Configuration

| File | Role |
|---|---|
| **[.env.example](../.env.example)** | Documented template for `.env`. Every `IFA_*` knob with rationale. Copy to `.env` once, edit to taste. |
| **[ifa/requirements.txt](../ifa/requirements.txt)** | Pinned Python deps. Note that `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` are only pulled on Windows/Linux for GPU Whisper; `scikit-learn`/`scipy` are range-pinned because exact versions break older Pythons. |

---

## Three concepts that thread through everything

### 1. The agent loop is the brain — tools are the limbs

Stage 0 of Ifa had an "intent classifier" that tried to map text to skills with regex/keywords. It was fragile and brittle. Stage 1 replaced that with **Ollama tool dispatch**: the LLM gets a list of tools (`get_time`, `set_reminder`, etc.) with JSON Schemas, and decides itself which to call. The agent loop is plumbing — it only handles message-passing, dispatch, and result delimiting. It contains zero domain logic.

If you want Ifa to do something new, **add a tool**. Look at `ifa/tools/time.py` (the simplest) for the pattern: declare a `Tool`, `register(TOOL)`, add to `register_all()`. The LLM will pick it up automatically once it's in the schema.

### 2. The `AgentContext` is dependency injection without a framework

The naive way to give tools access to `tts`, `db`, etc. is module-level singletons. That's what Stage 0 did, and it made testing hellish (every test had to reset globals). Stage 1 instead builds an `AgentContext` once in the orchestrator and passes it through `agent_turn` → `dispatch(tool_name, args, ctx)` → `handler(args, ctx)`. Tests construct their own contexts with mocks; production gets the real services.

### 3. Voice mode is a thread on top of text mode

The voice pipeline doesn't replace the text-mode loop — it produces input for it. `VoiceInput.get()` blocks on a `queue.Queue` that the daemon thread fills with transcribed text. From the orchestrator's view, voice and text are interchangeable: both implement `_InputMode` (just `get()` and `arm_followup()`). The complicated part is the voice thread itself; the orchestrator stays simple.

---

## Tests

Run the full suite from repo root:

```
PYTHONPATH=. venv/bin/python -m unittest discover ifa/tests
```

Notable test files:

| File | What it covers |
|---|---|
| **[test_tool_call_bench.py](../ifa/tests/test_tool_call_bench.py)** | NOT run in CI. Manual acceptance bench against live Ollama: 20 utterances with expected tool selections, prints per-tool accuracy. Gates the deletion of any further legacy paths. Run with `PYTHONPATH=. venv/bin/python -m ifa.tests.test_tool_call_bench`. |
| **[test_agent_loop.py](../ifa/tests/test_agent_loop.py)** | The agent_turn flow end-to-end with mocked Ollama. |
| **[test_tool_registry.py](../ifa/tests/test_tool_registry.py)** | Registry mechanics + JSON Schema validation. |
| **[test_orchestrator_stage1.py](../ifa/tests/test_orchestrator_stage1.py)** | Regression guards confirming the legacy detect_intent / extract_fact / extract_reminder paths stay deleted. |
| **[test_wake_word.py](../ifa/tests/test_wake_word.py)** | Wake-word listener with openWakeWord mocked + a real-onnxruntime integrity check on the bundled `.onnx`. |
| **[test_capture.py](../ifa/tests/test_capture.py)** | VAD-gated capture with Silero mocked. |
| **[test_voice_input.py](../ifa/tests/test_voice_input.py)** | The full voice-thread daemon loop with sounddevice + openWakeWord + Whisper all mocked. |

---

## Where to look when something is wrong

| Symptom | Suspect file(s) |
|---|---|
| Ollama health check fails on startup | `ifa/services/ollama_client.py:check_health`, `ifa/core/agent.py:MODEL` |
| Tool isn't being picked by the LLM | `ifa/core/agent.py` (system prompt), `ifa/tools/registry.py:as_ollama_schema`, the tool's own `description` and `parameters` |
| Reminder doesn't fire after restart | `ifa/core/orchestrator.py:resume_reminders`, `ifa/services/db.py` (schema), `ifa/skills/reminder.py:_reminder` |
| Voice mode hangs / no detection | `scripts/smoke_voice.py mic`, `scripts/smoke_voice.py record`, then `ifa/voice/wake_word.py`, `ifa/voice/input.py:_run_loop` |
| Self-feedback loop (Ifa transcribes herself) | `ifa/services/tts_service.py:is_speaking`, `ifa/voice/input.py:_wait_for_tts_silence`, `ifa/voice/input.py:_drain_stream` |
| Whisper returns empty / wrong text | `ifa/voice/stt.py:_get_model` (device + compute_type), `IFA_WHISPER_DEVICE` env, dump WAV via `IFA_VOICE_DEBUG_WAV` |

---

## Stage history (for context)

The codebase has been built incrementally. Each stage shipped to `main` with an acceptance gate.

| Stage | What landed | Plan doc |
|---|---|---|
| **Stage 1** | LLM tool dispatch (qwen2.5:7b-instruct), tool registry, n8n integration, fact memory | `docs/plans/2026-04-23-002-feat-stage1-llm-tool-dispatch-plan.md` |
| **Stage 2** | Voice input (wake word + VAD + Whisper), mute-during-TTS, follow-up window | `docs/plans/2026-04-23-003-feat-stage2-voice-input-plan.md` |
| **Stage 3** | Custom-trained `ifa.onnx` wake-word model | `docs/plans/2026-04-27-001-feat-custom-hey-ifa-wakeword-plan.md` |

Read those plans if you want the original problem framing, decisions, and tradeoff rationale that produced this code.

---

## Practical next steps for a new contributor

1. Read `ifa/main.py`. Then `ifa/core/orchestrator.py`'s `run()` function.
2. Read `ifa/core/agent.py`'s `agent_turn`. Trace what happens for the input `"what time is it"`.
3. Read one tool: `ifa/tools/time.py`. Then read `ifa/tools/registry.py:dispatch`.
4. If you're touching voice: read `ifa/voice/input.py`'s `_run_loop` method docstring carefully — it explains the mute/drain/event-coordination dance.
5. Run `PYTHONPATH=. venv/bin/python -m unittest discover ifa/tests` and read the names that pass. The test names are quasi-documentation.
6. Skim a recent commit (`git log --oneline -20`) — commits are usually scoped to one logical change with a thorough message; reading them in order is often easier than re-deriving the history from code alone.
