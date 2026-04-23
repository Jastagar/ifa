---
date: 2026-04-23
topic: conversation-and-tool-foundation
---

# Voice Conversation Loop + Tool-Calling Foundation

## Problem Frame

Ifa today is a text-in / voice-out REPL. To become the "AI companion that can do stuff" the user actually wants — with eventual n8n integration for action-taking — two things need to land together: a **voice conversation loop** (you speak, Ifa hears, Ifa speaks back, repeat) and a **tool-calling foundation** (the LLM reliably produces structured tool calls instead of free-text, dispatched through a unified registry that's ready for n8n webhooks).

Neither half delivers the goal alone. Voice without tools is a better interface to the same limited capabilities. Tools without voice is a worse interface to expanded capabilities. The two are bundled here because they share architectural touch points: the main orchestrator loop is replaced, the LLM is replaced, and the single `think() → speak()` path becomes a tool-dispatch loop.

A **previously-started neural TTS brainstorm** ([docs/brainstorms/2026-04-23-neural-tts-upgrade-requirements.md](2026-04-23-neural-tts-upgrade-requirements.md)) is explicitly deprioritized. The current subprocess TTS (`say`/PowerShell/espeak, shipped in commit `d7840fe`) stays. Voice quality is good enough; conversation and capability are the leverage.

## Requirements

### Voice input loop

- R1. Wake-word trigger via `openWakeWord` using a **built-in phrase** for v1 (likely `"hey jarvis"`). Model and `silero_vad.onnx` helper must be downloaded once via `openwakeword.utils.download_models()`; wake-word is instantiated with `inference_framework='onnx'` so it doesn't attempt to load `tflite-runtime` (which is not available on macOS). Custom `"hey ifa"` training is out of scope for v1.
- R2. After wake-word fires, capture microphone audio until `silero-vad` detects ~1.5 seconds of silence. Max utterance length capped at 30 seconds (hard cutoff).
- R3. Captured audio is transcribed via `faster-whisper` (already in `ifa/requirements.txt`) in batch mode — full-utterance-then-transcribe, not streaming. Mic capture via `sounddevice` must use `samplerate=16000, dtype='float32', channels=1` explicitly — sounddevice's default samplerate is the OS device rate (44100/48000Hz), which would produce silently garbled transcription when passed to faster-whisper (which assumes 16kHz).
- R4. The transcription replaces the `input("You: ")` call in [ifa/voice/input.py:12](../../ifa/voice/input.py#L12). The rest of the orchestrator loop sees the same string it does today.
- R5. After Ifa's TTS playback completes, the loop returns to wake-word listening state.
- R6. Text input mode (current `MODE = "text"` in [ifa/voice/input.py:5](../../ifa/voice/input.py#L5)) remains available as a fallback for when voice pipeline fails or for development work. Env var or config flag switches between modes.

### LLM upgrade (replaces Mistral)

- R7. Swap Mistral 7B for **qwen2.5:7b-instruct** (pulled as `qwen2.5:7b-instruct-q4_K_M` or equivalent instruct tag — the plain `qwen2.5:7b` tag resolves to the base model, which lacks the instruction-following needed for reliable tool calls). Runs locally via Ollama, pulled once per machine.
- R8. Use Ollama's native function-calling API rather than shelling out to `ollama run` via subprocess. The current [ifa/core/brain.py](../../ifa/core/brain.py) uses `subprocess.run(["ollama", "run", MODEL, prompt])`; this becomes an HTTP call to Ollama's `/api/chat` endpoint with `tools=[...]` in the payload.
- R9. `detect_intent()` and `handle_with_intent()` (the current primitive tool-routing) are **replaced** by the unified tool-call loop below — not preserved alongside. Individual skill logic (TimeSkill, ReminderSkill) is preserved as tool handlers.

### Tool-calling foundation

- R10. Unified tool registry: a single module (proposed: `ifa/tools/registry.py`) where each tool declares `name`, `description`, `parameters` (JSON Schema), and `handler(args) -> result`. Tools can be added or removed in one place.
- R11. **v1 tool set:** `get_time` (replaces TimeSkill), `set_reminder(task, seconds)` (replaces ReminderSkill), `call_n8n_workflow(workflow_name, payload)` (new), `read_file(path)` (new, sandboxed — see R19–R21).
- R12. Per-turn dispatch: LLM is called with the user's transcribed text + the tool registry. LLM responds with either a direct text answer OR a tool call. If tool call, the tool handler executes and the result is used to produce a final text response that's spoken to the user.
- R13. **v1 implements single-call dispatch:** exactly one tool call per user turn. The tool result is fed back into the LLM once to produce the user-facing reply, then the turn ends.
- R14. **v2 architecture is iterative** (documented in detail below but not shipped in v1): LLM → tool call → result → back to LLM → possibly more tool calls → final answer. Scaffolded in v1 by structuring the code as a loop with `max_iterations=1`. Migration to v2 is changing the cap + prompt guidance, not a rewrite.
- R15. Malformed tool calls from the LLM (invalid JSON, unknown tool name, bad args) trigger a single retry with an error-correction message back to the LLM. If retry also fails, Ifa speaks a graceful "I couldn't figure out how to do that" response. No crash.

### n8n integration

- R16. `call_n8n_workflow(workflow_name, payload)` maps `workflow_name` to a webhook URL via a config file (proposed: `ifa/config/n8n_workflows.yaml`). POSTs `payload` as JSON to the URL, returns the response as a structured tool result.
- R17. n8n config file example: `my_todo: https://n8n.local/webhook/abc123`. User edits this file to register workflows. **`ifa/config/n8n_workflows.yaml` must be added to `.gitignore` as part of the implementation** — webhook URLs contain secret tokens (the `abc123` path segment is bearer-token equivalent). A committed template file (`n8n_workflows.yaml.example` with placeholder values) is checked in; the real file is user-local. If per-workflow auth credentials are stored (R18), they must reference environment variables by name, not inline plaintext.
- R18. Auth: webhook-level (n8n's built-in webhook auth — basic auth or token header — configurable per workflow). No OAuth, no discovery protocol, no n8n API inspection — keep the integration surface tiny.

### `read_file` sandbox

- R19. `read_file(path)` only reads paths under a configurable sandbox root (default: `~/.ifa/sandbox`, created on first run if missing). Default is `~/.ifa/` not `~/Documents/ifa-sandbox` because on macOS `~/Documents/` is iCloud-synced by default — sandboxed files would be silently uploaded. First-run setup must verify the sandbox root is a real directory (not a pre-existing symlink) before trusting it as a boundary.
- R20. Path resolution is **realpath-before-check**: call `os.path.realpath(os.path.join(sandbox_root, user_path))` first, then verify the result is a descendant of `realpath(sandbox_root)`. Do not prefix-check the unresolved path. Any resolution error, failed sandbox-descendant check, or escaping symlink returns a structured tool error. TOCTOU between realpath and open is an accepted residual risk for a single-user local tool.
- R21. Max read size caps at 50KB — larger files return a truncation notice. Prevents a tool call from dumping gigabytes into LLM context.

### Shared scaffolding

- R22. Architecture supports the eventual v2 iterative loop without refactor — tool results feed back into context, `max_iterations` is a config constant. See "v2 Iterative Loop (Deferred Implementation)" below for the detailed design that v1 must not foreclose.
- R23. Existing `ifa/core/memory.py` continues to hold recent conversation context. Tool results become part of memory so the LLM can refer back to them in later turns (e.g., "what file did you read earlier?"). Deeper memory work is out of scope.

## Success Criteria

- Saying `"hey jarvis, what time is it?"` into the mic produces an audible spoken response with the current time within ~3-5 seconds (single-turn target; see open latency concern in Outstanding Questions).
- Saying `"hey jarvis, remind me to stretch in 30 seconds"` sets the reminder, which fires audibly 30s later.
- Saying `"hey jarvis, trigger my home_summary workflow"` (with a matching entry in `n8n_workflows.yaml`) POSTs to n8n, gets a response, and Ifa speaks a summary of the response. n8n call latency is additive to the baseline — workflows expected to complete within the HTTP timeout configured per-workflow (see R18).
- Saying `"hey jarvis, read the shopping list"` (with `shopping.txt` in the sandbox dir) reads the file and speaks a summary.
- Text-mode fallback still works (`MODE=text` in a config) — for development and voice-pipeline-broken debugging.
- LLM swap is zero-config for the user on either machine (qwen2.5:7b just works once pulled via `ollama pull qwen2.5:7b`).

## Scope Boundaries

- **Interruption support is not in scope.** User cannot cut Ifa off mid-speech in v1. Wait for playback to finish, then trigger the wake word again.
- **Multi-user / voice ID is not in scope.** Ifa treats any voice reaching the mic as the user.
- **Custom `"hey ifa"` wake-word training is not in scope for v1.** Follow-up work once the loop is proven.
- **Neural TTS is not in scope.** Parked in the previous brainstorm; current subprocess TTS is retained.
- **Iterative tool loop IMPLEMENTATION is not in scope for v1.** Scaffolded but `max_iterations=1` is hardcoded. The v2 design is documented below.
- **Deeper memory / retrieval is not in scope.** Existing `memory.py` behavior preserved.
- **n8n trigger-to-Ifa (inbound) is not in scope.** Only Ifa-to-n8n (outbound via webhook POST).
- **Streaming STT is not in scope.** Batch transcription after end-of-turn detection.

### Deferred to Separate Tasks

- **Custom `"Hey Ifa"` wake word** — train via openWakeWord's custom-training pipeline (~3-4 hours effort including synthetic data generation). Sits in a follow-up brainstorm/plan once v1 ships.
- **Neural expressive TTS (Parler-TTS)** — already brainstormed at [docs/brainstorms/2026-04-23-neural-tts-upgrade-requirements.md](2026-04-23-neural-tts-upgrade-requirements.md); pick up after conversation + tools are stable.
- **Output-device picker** — earliest brainstorm at [docs/brainstorms/2026-04-23-output-device-picker-requirements.md](2026-04-23-output-device-picker-requirements.md); fully superseded by "works on OS default" for now.
- **Iterative tool loop (v2)** — designed below, implemented later. No user-visible behavior change from v1 initially; first value arrives when we add a tool that meaningfully benefits from chained calls.

## Key Decisions

- **Wake word built-in > custom for v1**: ships fastest with openWakeWord's included phrases. Validates the voice loop end-to-end before investing in custom wake-word training. Ifa's "name" can be `Hey Ifa` in spirit even if the trigger phrase is `Hey Jarvis` for v1.
- **qwen2.5:7b over Mistral**: native tool-calling support. Mistral 7B would force continuous JSON-repair work (the existing `extract_reminder()` already demonstrates this brittleness). The LLM swap is a one-line change in Ollama config; the leverage is enormous.
- **Ollama HTTP API over subprocess CLI**: structured-output support requires the HTTP API (`format: "json"` or `tools=[...]`). `subprocess.run(["ollama", "run", ...])` cannot express structured calls cleanly. Migration is straightforward — `ollama` Python client is already available, or httpx (already a dep) against Ollama's REST endpoint.
- **Single-call v1 + iterative v2 scaffolded**: ship sooner, architect for later. v1 delivers real value (Ifa "does one thing" per voice turn via n8n). v2 is a loop-cap change + prompt update.
- **Unified tool registry > scattered skill dispatch**: one place to define tools, one dispatch path, easy to add/remove. Current skill pattern is fine for 2 skills; we're growing to 4 in v1 and more later.
- **n8n via webhook POST**: simplest possible integration. User manages workflows in n8n's UI, registers webhook URLs with friendly names in a YAML file. No bidirectional sync, no API discovery, no protocol invention.
- **read_file sandboxed**: security by default. Solo-user tool but the LLM is a confused deputy — a prompt injection via n8n response could trick the LLM into calling `read_file("/etc/passwd")` or `read_file("~/.ssh/id_rsa")` otherwise. Sandbox limits blast radius.
- **Voice text-mode fallback retained (R6)**: when something breaks in the voice pipeline (bad mic, no wake word model downloaded, etc.), you can still type. Essential during development and debugging.
- **Neural TTS deferred**: voice is the interface, but the voice *already works*. Making it more expressive is user-delight polish; making Ifa *hear* and *act* is the capability jump. Sequencing matters.

## v2 Iterative Loop (Deferred Implementation, Designed Now)

Per the user's request to detail the iterative design even while shipping single-call. v1 must structurally accommodate v2 without refactor.

**Architecture:**

```
function agent_turn(user_text):
  messages = [system_prompt, ...conversation_history, {role: user, content: user_text}]
  for iteration in 1..MAX_ITERATIONS:
    response = ollama.chat(model=qwen25, messages=messages, tools=tool_registry.as_ollama_schema())
    if response has tool_calls:
      for each tool_call:
        result = tool_registry.dispatch(tool_call.name, tool_call.args)
        messages.append({role: "tool", tool_name: tool_call.function.name, content: result})
      # NOTE: Ollama's /api/chat expects the field `tool_name`, not `name`, for
      # tool-role messages. Using `name` causes the LLM to see unattributed results.
      continue  # LLM may want to call more tools or respond
    else:
      # LLM chose to respond with text — terminal
      return response.content
  # max_iterations exhausted
  return "I got stuck trying to work that out. Can you simplify?"
```

**v1 vs v2 diff:**
- v1: `MAX_ITERATIONS = 1` as a constant in `ifa/tools/registry.py` (or wherever the loop lives). If LLM returns a tool call, execute it, feed result back, get terminal text response. If LLM returns another tool call after the result, we currently force terminal by rejecting it with "tool call limit reached" — the LLM has to respond with text in this second pass.
- v2: `MAX_ITERATIONS = 5` (or similar). Same code path, just allows chains. System prompt shifts from "call at most one tool then respond" to "call tools as needed, then respond."

**v1 carries all v2 scaffolding:**
- Tool registry shape (name, description, JSON schema parameters, handler function).
- Messages-list-with-tool-role format matching Ollama's function-calling API.
- Tool result feedback into conversation context (required for v1's single-call too — LLM needs to see the tool result to formulate a reply).
- Error handling for malformed tool calls (R15).
- `max_iterations` as a single constant to change.

**What v2 explicitly adds that v1 does NOT have:**
- Safety: per-turn timeout cap (30s?) to prevent runaway tool-call chains.
- Observability: log each iteration with tool name, args, result, so debugging chains is possible.
- Prompt tuning: the system prompt must actively encourage multi-step reasoning once iteration is enabled (qwen2.5 supports this natively but benefits from guidance).
- Potentially: tool-side permissioning / confirmation for dangerous tools. Irrelevant while all tools are local or n8n-mediated, becomes relevant when tools can spend money or trigger real-world side effects.

## Dependencies / Assumptions

- `openWakeWord` is **not** yet in `ifa/requirements.txt` — new dependency, pure Python + ONNX, works on macOS and Windows. Built-in wake-word models are included in the package.
- **VAD** — `faster-whisper` already bundles `silero_vad_v6.onnx` in its assets directory and exposes `faster_whisper.vad.SileroVADModel` backed purely by `onnxruntime` (already installed). Using this internal model avoids a new dependency entirely. Planning should treat `pip install silero-vad` as the fallback, with awareness that it pulls `torch>=1.12` + `torchaudio` (~600MB) — directly contradicting the project's no-torch posture. Prefer the faster-whisper internal path; accept torch only if that path turns out unstable in practice.
- `faster-whisper` IS already in `ifa/requirements.txt` (v1.2.1). `ctranslate2` (its runtime) is also present. Works cross-platform.
- `sounddevice` IS already a dep — used for mic capture.
- `httpx` IS already a dep — used for Ollama HTTP API calls.
- `PyYAML` IS already a dep — used for n8n workflow config file.
- **n8n is installed and reachable by the user** at the URLs they configure. If not, `call_n8n_workflow` tool errors gracefully. n8n setup is user's responsibility, not Ifa's.
- **qwen2.5:7b is available via Ollama** (verified — `ollama pull qwen2.5:7b` works). Supports native tool calling as of Ollama 0.3.x.
- **User is willing to `ollama pull qwen2.5:7b`** once per machine. Not automated.

## Outstanding Questions

### Resolve Before Planning

- **[Affects R9][Scope] Fate of `extract_fact()` and the `facts` SQLite table.** Current orchestrator calls `extract_fact()` on every user turn via a subprocess LLM call and `brain.think()` reads up to 5 facts from the table into every prompt. R9 names `detect_intent` and `handle_with_intent` as replaced but is silent on `extract_fact`. Three resolutions are acceptable, but one must be picked before planning: (a) preserve `extract_fact` as a post-response hook in the new agent loop (behavioral continuity), (b) replace with an LLM-callable `remember_fact(fact)` tool (fits new model, user-initiated), or (c) drop entirely and accept the regression.
- **[Affects R10][Technical] Tool registry location + skill migration.** `ifa/tools/registry.py` is proposed; unresolved whether `ifa/skills/*.py` (TimeSkill, ReminderSkill) are migrated into `ifa/tools/*.py` or wrapped in place as tool adapters. Each path has real file-churn implications — decide now so planning units are sized correctly.
- **[Affects R18][Technical] n8n per-workflow auth structure.** Config file shape (flat name→URL vs richer per-workflow auth/timeout block) must be fixed before R17 is implemented, because changing the YAML schema later breaks already-configured user workflows. A flat map cannot represent per-workflow auth, which R18 requires.
- **[Affects R11, R19][Product] Is `read_file` actually in v1 scope?** Product-lens + scope-guardian consensus: none of the four success criteria load-bear on `read_file` except the last one, and the user's long-term goal (n8n action-taking companion) doesn't require it. Three dedicated requirements (R19–R21) plus sandbox + symlink-resolution plumbing is disproportionate to demonstration value. Decide: keep it, or defer to a later iteration and trim to 3 tools.
- **[Affects R5, R1][Architecture] Wake-word coordination during TTS playback.** Multiple reviewers converge on this: the wake-word listener is continuously running, Ifa speaks via TTS, openWakeWord hears Ifa's own voice, potentially triggers on her own utterance, and silero-vad captures residual audio into garbage LLM turns. Additionally reminder daemon threads fire TTS concurrently with the main loop. Must specify: does the mic/wake-word listener mute or pause during any TTS playback (main response + reminder daemon threads)? What coordinates this across threads? No acoustic echo cancellation is specified; something equivalent is needed.

### Deferred to Implementation

- [Affects R1][Technical] Which specific openWakeWord built-in phrase for v1? `"hey jarvis"` is most battle-tested. `"computer"` also works. Pick during Unit 1 based on false-activation behavior in the user's actual environment.
- [Affects R3][Technical] Which faster-whisper model size? `base` (fast, okay accuracy), `small` (balanced), `medium` (slower, better), `large-v3` (slowest, best). Benchmark on both target machines with 5-10 real user utterances.
- [Affects R7][Technical] qwen2.5:7b quantization? Ollama defaults to Q4_K_M. Acceptable for v1; test accuracy against tool-calling benchmark before considering Q5 or Q8.
- [Affects R8][Technical] Use Ollama's official Python client library, or httpx direct against the REST endpoint? The official client is more idiomatic but adds a dep; httpx is already in. Lean toward httpx.
- [Affects R10][Technical] Where does the tool registry live structurally — `ifa/tools/registry.py` with handlers as sibling modules (`ifa/tools/time.py`, `ifa/tools/reminder.py`, etc.)? Migrate existing `ifa/skills/*` into the new location? Or keep skills and wrap them as tool adapters? Decide during planning — has ripple effects but no external impact.
- [Affects R16][Technical] YAML schema for `n8n_workflows.yaml`. Simplest: flat map of name → URL. Richer: per-workflow auth method, timeout, expected response shape. Start flat, extend later.
- [Affects R19][Technical] Sandbox root default path. `~/Documents/ifa-sandbox` is a suggestion — user-home-scoped feels right but platform conventions differ (macOS `~/Documents`, Windows `%USERPROFILE%\Documents`, Linux XDG). Use `pathlib.Path.home() / "Documents" / "ifa-sandbox"` cross-platform.
- [Affects R2][Needs research] silero-vad end-of-turn silence threshold — 1.5s is a guess. Parameterize and tune based on user testing. Too short = cuts off mid-thought, too long = feels sluggish.
- [Affects R11][Needs research] Will qwen2.5:7b reliably select the right tool for natural phrasings ("remind me to..." → `set_reminder`, "what time is it" → `get_time`)? Unit 2 should include a small bench of expected utterances to confirm before declaring the tool layer done.
- [Affects R9][Scope] How much of `ifa/core/brain.py` survives the rewrite? `think()`, `detect_intent()`, `extract_reminder()`, `extract_fact()` all become obsolete if tool-calling handles everything. Probably `brain.py` is deleted or minimized, replaced by `tools/` + `agent.py`. Plan this as a distinct implementation unit.

## Next Steps

-> Resume `/ce:brainstorm` to resolve the 5 **Resolve Before Planning** items. Several can be answered with short decisions rather than deep investigation. Once resolved, `-> /ce:plan`.
