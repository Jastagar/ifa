# Agent loop, tools and memory

How one turn turns your words into tool calls and a spoken reply, and the guard rails around a small local model.

**Files:** [ifa/core/agent_stream.py](../../ifa/core/agent_stream.py), [ifa/tools/registry.py](../../ifa/tools/registry.py), [ifa/tools/](../../ifa/tools/), [ifa/core/memory.py](../../ifa/core/memory.py), [ifa/services/memory_store.py](../../ifa/services/memory_store.py), [ifa/services/prompts.py](../../ifa/services/prompts.py), [ifa/core/orchestrator.py](../../ifa/core/orchestrator.py)

---

## The loop (ReAct)

```
user text ─► build messages: system prompt + last 4 turns + user text (+ unread-notification hint)
                │
                ▼
        ┌─► stream_chat(model, messages, tools)  ── sentences stream to TTS as they complete
        │       │
        │       ├─ no tool calls ──► reply is final ─► remember the turn ─► done
        │       │
        │       └─ tool calls ─► dispatch every call (one result per call, errors included)
        │                        wrap each result in nonce markers, append to messages
        └──────── up to MAX_TOOL_HOPS = 3 rounds; the 4th call has no tools, so the model must answer
```

**Pattern: ReAct** (reason, act, observe, repeat). Before this, Ifa could make one tool call per turn, so "look at my screen and remind me to fix the error" couldn't work. Now the model can act, see the result, and act again.

**Decisions**
- **3 hops, then tools are withheld.** A budget keeps a confused model from looping forever. Withholding tools on the last round forces an answer from what it already has.
- **N calls get N results**, even when one fails. The old loop stopped at the first error and left the model with fewer results than calls, which broke the protocol.
- **Errors are data, not exceptions.** `registry.dispatch` validates arguments with JSON Schema and turns any failure into an `ERROR: ...` result. The model can then apologise, retry, or try another tool.
- **Terminal tools** (`return_direct=True`, e.g. `read_notifications`): the tool's own result is spoken as-is and the turn ends. The model gets no chance to add "shall I reply?".

## Trust boundary: tool results are data

Tool results (notification text, web pages, command output) are written by someone else. Each turn gets a random nonce, and each result is wrapped as `<TOOL_RESULT_xxxx_START tool=NAME> ... <TOOL_RESULT_xxxx_END>`. The system prompt says content between the markers is data, never instructions. Because the nonce is random per turn, a message can't fake the closing marker.

## Guards against a small model's habits

A 9B model gets tool calling right most of the time, but it fails in predictable ways. The rule we settled on: **steer with the prompt, enforce with code.**

| Failure seen live | Guard | Where |
|---|---|---|
| The model typed a fake `TOOL_RESULT_...` block instead of calling the tool | Any reply containing the marker stops speaking and gets one retry with a correction | `TOOL_RESULT_MARKER`, `FABRICATION_CORRECTION` |
| "Playing X" with no tool call (after a first song, the history taught it to answer "play X" with text) | If no tool has run yet in the turn and the **first sentence** claims an action ("Playing…", "Opened…", "X's open"), it isn't spoken and the model gets one correction. A second identical claim is spoken, so chat like "Running is good for you" isn't blocked. | `ACTION_CLAIM`, `ACTION_CLAIM_CORRECTION` |
| Malformed tool call (arguments not an object) | One retry with the reason, then give up | `_tool_call_errors` |
| A stranger saying "yes" to a command | Speaker verification + the run_command confirmation gate | [speaker-verification.md](speaker-verification.md), [tools.md](tools.md) |

## Short-term memory: whole turns, tool calls included

`Memory` keeps the **last 4 turns**. Each turn is stored as user text, then every tool call with a one-line result, then the final reply.

**Why:** memory used to keep only the user's words and the final reply. After the first song, the history read `play tum hi ho → "Playing Tum Hi Ho"`, and the model copied that pattern and stopped calling the tool. **The conversation history is a few-shot prompt**, so it has to show real tool calls. It also used to send the last 5 *messages*, which could start mid-turn.

Tool results in memory are cut to their first line (160 characters) to keep prompts small and to avoid carrying large amounts of third-party text forward.

## Long-term memory: topics

Set up in another session (commit `99ed574`). Memories live in `ifa.db`, one per topic, each tagged `progress`, `preference`, `instruction` or `fact`. `remember` replaces a topic (or a close variant), `recall` fuzzy-matches a question, `forget` deletes. Preferences and instructions go into every prompt. For the rest, only the topic names do, and Ifa recalls them on demand. **Why:** the old `memory.md` only grew and was pasted whole into every prompt. The launcher has a Memory card to edit them.

## Prompts and persona

- Prompts (persona, notification urgency, notification summary) live in `ifa/services/prompts.py`. They can be edited in the launcher and are saved to `prompts.json`, which applies on the next turn.
- **Persona:** one or two short sentences, always English (with an example reply for Hinglish input, because the bare rule was ignored), spoken text only, no closing offers like "let me know if you need anything else", and after a tool, a few words of confirmation.
  - **Measured on qwen3.5:9b:** replies went from 2–5 sentences to 1–3, the closing offers disappeared, and Hinglish questions got English replies 11/12 times (0/1 before).
  - **Lesson:** the old persona literally said "4–5 sentences". Its sentences were also joined without spaces.
- **Prompt lessons from this project:**
  1. **Describe what to do.** "Never reply to notifications" made the model skip the notification tool entirely.
  2. **Give an example.** "Reply in English" was ignored until it came with an example reply.
  3. **When the prompt keeps failing, change the structure.** Terminal tools, code guards and enums are more reliable than more rules.

## Tool selection: descriptions matter

The `open_app` description used to tell the model to "ask for confirmation first", and its strings were concatenated without spaces. "Open Spotify and play X" sometimes played through the API without opening the app.

The fix was to say so explicitly: opening and controlling are separate steps, call `open_app` first, and `spotify_control` never opens the app. **Measured:** 15/16 → 20/20 on the full loop.

Tool schemas are about **2,000 of the ~2,450 prompt tokens** on every turn, so long `MUST` blocks in descriptions cost latency on every request.

## Tools (current)

`get_time`, `set_reminder`, `remember`, `recall`, `forget`, `call_n8n_workflow`, `switch_mode`, `open_app`, `open_website`, `spotify_control`, `whats_app`, `analyze_screen`, `launch_game`, `read_notifications`, `run_command`, `end_conversation`, `restart_ifa`. Details of the newer ones: [tools.md](tools.md).

## Conversation control (other session, `ac96f8a`)

- A message that's only a goodbye ("that's all", "bas", "ho gaya") ends the turn with no model call and no follow-up listening.
- `restart_ifa` exits with code **75**, and `python -m ifa.main` runs a small supervisor that restarts on that code. That's how restarts pick up code and `.env` changes.
