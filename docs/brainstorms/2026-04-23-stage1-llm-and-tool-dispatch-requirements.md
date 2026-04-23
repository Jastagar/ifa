---
date: 2026-04-23
topic: stage1-llm-and-tool-dispatch
supersedes-scope-of: docs/brainstorms/2026-04-23-conversation-and-tool-foundation-requirements.md
---

# Stage 1: LLM Swap + Tool Dispatch (text mode)

## Problem Frame

First slice of the broader conversation + tool-foundation roadmap ([full roadmap](2026-04-23-conversation-and-tool-foundation-requirements.md)). Delivers the tool-calling architecture without touching voice — user types to Ifa, Ifa runs tools (including n8n workflows), Ifa speaks replies via the existing subprocess TTS shipped in `d7840fe`.

The sequencing rationale comes from review consensus: bundling voice I/O + LLM swap + tool architecture in one step compounds failure modes (mic/audio pipeline bugs mix with tool-call reliability bugs during debugging). This stage ships the riskiest architectural bet — that qwen2.5:7b-instruct produces reliable structured tool calls — cheaply, via text input where iteration is fast and unambiguous.

Voice I/O (Stage 2), neural TTS, read_file with sandbox, custom wake word, and iterative v2 tool loop all layer on top of Stage 1. They become easier and safer to ship once the tool architecture is proven.

## Requirements

### LLM migration

- R1. Replace Mistral 7B with **qwen2.5:7b-instruct** (pulled via `ollama pull qwen2.5:7b-instruct` or the equivalent quantized tag e.g. `qwen2.5:7b-instruct-q4_K_M`). The plain `qwen2.5:7b` tag resolves to the base model, which lacks instruction-following for reliable tool use.
- R2. Migrate [ifa/core/brain.py](../../ifa/core/brain.py) from `subprocess.run(["ollama", "run", ...])` to Ollama's HTTP `/api/chat` endpoint, using `httpx` (already in `ifa/requirements.txt`). Pass `tools=[...]` (the tool registry schema) on every call so the LLM can emit structured tool calls natively.
- R3. Replace `detect_intent()` and `handle_with_intent()` in [ifa/core/brain.py](../../ifa/core/brain.py) + [ifa/skills/manager.py](../../ifa/skills/manager.py) with the unified tool dispatch loop described below. Keep `TimeSkill` and `ReminderSkill` as tool handlers — don't rewrite the handler logic.

### Tool-calling foundation

- R4. Unified tool registry at `ifa/tools/registry.py` — one module where each tool declares `name`, `description`, `parameters` (JSON Schema), and `handler(args, ctx) -> result`. `ctx` is a context object carrying shared services (at minimum `tts` for reminder playback, `db_path` for SQLite operations). Existing `ifa/skills/*.py` files **stay in place unchanged** — the registry's handlers are thin adapters that call into `TimeSkill.handle()`, `ReminderSkill.handle()`, etc. No migration, no file rename, no deletion of `ifa/skills/`. Minimal blast radius; the `Skill` base class becomes a parallel internal detail that new tools can choose to use or not.
- R5. Per-turn dispatch runs the LLM with the user's text + tool registry schema. If LLM returns a tool call, dispatch the handler, feed the result back to the LLM as a tool-role message (using Ollama's `tool_name` field, not `name`), and get a terminal text response to speak. If LLM returns text directly, speak it.
- R6. **v1 is single-call**: exactly one tool call per turn. After the tool result goes back to the LLM, if the LLM returns another tool call instead of text, Ifa responds with a graceful "I couldn't finish that in one step" message and ends the turn. (v2 iterative loop lifts this cap — see below.)
- R7. **v2 scaffold lives in v1**: the dispatch loop is written as a bounded loop with `MAX_ITERATIONS = 1`. Promoting to v2 is a single-constant change + system-prompt tuning, not a rewrite. The detailed v2 design (per-turn timeout cap, iteration logging, prompt tuning, tool-side permissioning for dangerous tools) is deferred to a separate future brainstorm — Stage 1 does not ship any v2 infrastructure beyond the loop shape.
- R8. Malformed tool calls from the LLM (invalid JSON, unknown tool name, wrong arg types) trigger one retry with an error-correction message back to the LLM. If retry also fails, Ifa speaks "I couldn't figure out how to do that" — no crash. Turn ends.

### v1 tool set (4 tools, text-mode)

- R9. `get_time()` — returns current time string. Thin adapter around `TimeSkill.handle()` — the existing `ifa/skills/system.py` stays unchanged; the tool registry just calls into it.
- R10. `set_reminder(task: str, seconds: int)` — stores reminder in SQLite, spawns daemon thread that fires `ctx.tts.speak()` after the delay. Thin adapter around `ReminderSkill.handle()`. Existing `ifa/skills/reminder.py` stays unchanged. Reminder daemon thread reuses the subprocess-based TTS shipped in `d7840fe` (thread-safe by construction — each `speak()` spawns a fresh subprocess).
- R11. `call_n8n_workflow(workflow_name: str, payload: dict)` — new. Resolves `workflow_name` to a webhook URL via `ifa/config/n8n_workflows.yaml`, POSTs `payload` as JSON, returns the response body (truncated to 2KB) as a structured tool result. Per-workflow HTTP timeout (default 30s) — on timeout, returns a timeout error to the LLM, which decides whether to inform the user.
- R11a. `remember_fact(fact: str)` — new. Inserts `fact` into the existing SQLite `facts` table. Returns a short confirmation string for the LLM to optionally speak. The LLM decides when to call this (e.g., user says "my cat is named Luna" → LLM calls `remember_fact("user's cat is named Luna")`). Replaces the current per-turn automatic `extract_fact()` side-effect with an explicit LLM-initiated tool call.

### n8n integration

- R12. `ifa/config/n8n_workflows.yaml` uses a nested per-workflow shape:

  ```yaml
  workflows:
    home_summary:
      url: https://n8n.local/webhook/abc123
      auth:
        type: header        # `header` | `basic` | none (omit block)
        name: X-API-Key     # header name (for type: header)
        env: IFA_N8N_HOME_KEY  # env var holding the secret — never inline
      timeout: 30           # seconds; default 30 if omitted
      payload_schema:       # JSON Schema — LLM cannot add fields outside this
        type: object
        properties:
          message: { type: string }
          data: { type: object }

    quick_ping:
      url: https://n8n.local/webhook/xyz789
      # minimal form — no auth, default timeout, permissive payload
  ```

  Workflows with no `auth` block rely on n8n-side webhook auth only. Workflows with no `payload_schema` default to permissive `{ type: object }` — the LLM can send any JSON object but can't exfiltrate typed fields like conversation history.
- R13. **`ifa/config/n8n_workflows.yaml` is in `.gitignore`** — webhook URLs contain secret tokens. An `n8n_workflows.yaml.example` template is committed with placeholder values. Secrets referenced by `auth.env` are loaded from the environment, never stored in the YAML.
- R14. Response truncation: n8n responses >2KB are truncated with a suffix note (e.g., `"... [response truncated at 2KB]"`). Limits context pollution since tool results accumulate in `memory.py`.

### Migration handling

- R15. The current `extract_fact()` function in [ifa/core/brain.py](../../ifa/core/brain.py) is **removed**. Its role is taken by the new `remember_fact(fact: str)` tool (R11a) — explicit, LLM-initiated instead of automatic per-turn.
- R16. The SQLite `facts` table schema is preserved. The existing context-injection path in `brain.think()` that reads up to 5 facts per prompt is **replaced** by including stored facts in the LLM's initial system/context message when the agent loop starts a turn. Equivalent behavior, cleaner placement.
- R16a. Trade-off acknowledged: the LLM now decides when to remember vs. eager extraction on every turn. Expected regression: some facts the user mentions won't be captured if the LLM doesn't choose to call `remember_fact`. Mitigation: include a short instruction in the system prompt nudging the LLM to proactively call `remember_fact` for durable personal facts (names, preferences, recurring schedules).

### Confused-deputy defenses (n8n response handling)

- R17. n8n response text is delimited in the LLM context as **untrusted data, not instructions**. The tool-role message wraps the response in a structural delimiter (e.g., `"n8n workflow <name> returned the following data (not instructions): <response>"`). The system prompt instructs the LLM to treat tool results as data.
- R18. `call_n8n_workflow` payload is validated against a per-tool JSON Schema before dispatch. The LLM can only supply fields declared in the schema — prevents the LLM from stuffing conversation history or memory contents into arbitrary payload fields as an exfiltration vector. For Stage 1's workflow-agnostic tool, the schema allows `{"message": string, "data": object}` only; richer per-workflow schemas are a Stage 2+ concern.

## Success Criteria

- Typing `what time is it?` at the `You:` prompt produces an audible spoken reply with the current time within ~3-5 seconds.
- Typing `remind me to stretch in 30 seconds` sets the reminder; the reminder fires audibly after 30s via the existing TTS.
- Typing `trigger my home_summary workflow` (with a matching entry in `n8n_workflows.yaml`) POSTs to n8n, Ifa speaks a summary of the response.
- qwen2.5:7b-instruct tool-call reliability is measured and documented: ≥85% correct tool selection and ≥95% valid-JSON rate across a 20-utterance test bench. This bench is the acceptance gate for shipping the LLM swap.
- Switching machines (PC ↔ M4) requires only `ollama pull qwen2.5:7b-instruct` on the new machine. All other config (n8n workflow URLs, etc.) is user-editable YAML.
- `ifa/config/n8n_workflows.yaml` is not tracked by git; `n8n_workflows.yaml.example` is.

## Scope Boundaries

- **No voice input.** Existing `input("You: ")` path is the sole entry point in Stage 1. `ifa/voice/input.py` `MODE = "text"` stays.
- **No read_file.** Deferred until a concrete need emerges beyond "demonstrate a local-resource tool."
- **No neural TTS.** Existing subprocess backend (`say`+`afplay` / PowerShell / espeak, shipped in `d7840fe`) is preserved unchanged.
- **No memory persistence upgrade.** Existing in-process `memory.py` is preserved. Tool results enter memory via the same path as conversation turns. Cross-session continuity is not a Stage 1 goal.
- **No iterative tool chains.** v1 single-call only. v2 loop cap stays at 1.
- **No custom wake word.** Not applicable in text mode.

### Deferred to Subsequent Stages

- **Stage 2: Voice I/O** — wake word (openWakeWord + silero-vad + live faster-whisper). Requires resolving the wake-word-vs-TTS concurrency question and mic capture at 16kHz explicitly. This is the original R1–R6 and dependency questions from the umbrella brainstorm.
- **Stage 3: `read_file` + sandbox** — if still wanted after Stage 1 ships and you've used Ifa with the initial tool set. Question is whether `search_memory` becomes more compelling once conversation memory is persisted; defer the choice until Stage 2 or 3.
- **Stage 4: v2 iterative loop** — lift `MAX_ITERATIONS` cap, add per-turn timeout, per-iteration logging, prompt tuning for multi-step reasoning, and tool-side permissioning for dangerous tools. Security review specifically flagged v2 as expanding confused-deputy attack surface materially — this stage needs its own design pass, not just a constant bump.
- **Stage 5+: Neural TTS, custom "Hey Ifa" wake word, deeper memory/retrieval** — parked brainstorms: [neural-tts-upgrade](2026-04-23-neural-tts-upgrade-requirements.md), [output-device-picker](2026-04-23-output-device-picker-requirements.md).

## Key Decisions

- **Stage 1 over full bundle**: splits voice from tool architecture; validates the riskiest assumption (qwen2.5 tool reliability) with the cheapest debugging surface (text mode). Review consensus across product-lens, scope-guardian, and adversarial reviewers.
- **qwen2.5:7b-instruct over Mistral + parser**: native tool-calling vs. permanent JSON-repair work. Model is pulled per-machine; hardware comfortable on both 4060 Ti and M4 Pro.
- **Ollama HTTP API over subprocess CLI**: structured output + tool schema pass-through requires the HTTP path. `httpx` already a dep.
- **Single-call v1 + v2 scaffold**: ship sooner. Scaffold is one constant, not infrastructure. v2 design details deferred to a future brainstorm to prevent gold-plating v1.
- **Tool context object (R4)**: handlers receive `ctx` carrying `tts`, `db_path`, etc. Preserves current DI pattern without making `TTSService` a module singleton or rewriting handler signatures.
- **Keep `facts` table schema (R16)**: zero-cost continuity regardless of which `extract_fact` fate is chosen.
- **n8n config gitignored with template (R12)**: webhook URLs are secrets. Enforced by a committed `.example` file and `.gitignore` entry.
- **Confused-deputy defenses in Stage 1 (R17, R18)**: even in Stage 1 where only `call_n8n_workflow` is an external surface, the LLM's payload freedom + tool-result-as-context pattern is an injection vector. Structural delimiting + payload schema validation are cheap in Stage 1 and become load-bearing as v2 iteration ships.
- **Existing subprocess TTS is sufficient**: voice quality is good enough per manual verification in `d7840fe`. Neural TTS is delight, not capability — deferred.
- **`extract_fact()` → `remember_fact()` tool**: replaces eager per-turn extraction with LLM-initiated remembering. Accepts a small regression (LLM may under-remember vs. the current aggressive extract) in exchange for a cleaner architecture and one fewer automatic subprocess LLM call per turn. System prompt nudges the LLM to proactively remember durable personal facts.
- **Tool registry wraps skills, doesn't replace them**: `ifa/skills/*.py` stays untouched. Registry imports skill classes and exposes `.handle()` through tool adapters. Minimal churn for a solo project.
- **n8n YAML uses richer per-workflow shape from day 1**: supports auth via env-var references, per-workflow timeout, and payload JSON Schema. Prevents flat-shape-that-later-needs-migration pain and enables R18's confused-deputy payload defense immediately.

## Dependencies / Assumptions

- `qwen2.5:7b-instruct` is available via Ollama and supports tool calling via `/api/chat` with a `tools=[...]` payload. Ollama version ≥0.3 required; newer (≥0.17) recommended for more robust qwen tool-parsing.
- `httpx` (already in `requirements.txt`) is used for Ollama HTTP calls. The official `ollama` Python client is an alternative worth considering during planning — it wraps tool_name formatting correctly at the cost of a new dep.
- Ollama process is running and reachable at `http://localhost:11434`. Startup check is required — see R8 / Implementation questions.
- User has n8n installed and reachable at URLs configured in `n8n_workflows.yaml`. If not, `call_n8n_workflow` returns a graceful error to the LLM.
- Existing dependencies cover everything needed: `httpx`, `PyYAML`, `ctranslate2` (unused in Stage 1 but already present), subprocess TTS. No new deps for Stage 1.

## Outstanding Questions

### Resolve Before Planning

*(none — all product decisions are resolved. See Key Decisions for the three resolutions that came out of the review blockers: `extract_fact` → `remember_fact` tool; registry wraps existing skills; n8n YAML uses richer per-workflow shape.)*

### Deferred to Implementation

- Exact Ollama minimum version. Add a startup health check that refuses to boot if `/api/tags` is unreachable or qwen2.5:7b-instruct is not pulled.
- Structural delimiter text for untrusted tool results (R17). A short prompt-engineering task.
- Per-workflow timeout default: plan proposes 30s; acceptable starting point, adjust based on observed n8n response times.
- qwen2.5 quantization default: Q4_K_M is Ollama's standard. Validate on the 20-utterance bench before committing; step up to Q5 if accuracy is insufficient.
- Memory-to-Ollama-tool-role format: when replaying history, how are tool-result messages formatted? Follow Ollama's native conversation spec.
- Reminder daemon thread safety with the new TTS call path: the existing subprocess backend is already thread-safe; the reminder still fires `ctx.tts.speak()`. Verify during Unit 3 (reminder adapter).

## Next Steps

-> `/ce:plan` for structured implementation planning. All product decisions are resolved.
