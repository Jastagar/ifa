---
date: 2026-04-23
topic: neural-tts-upgrade
---

# Neural TTS Upgrade (Parler-TTS)

## Problem Frame

Ifa speaks through the OS's native voice (`say` on macOS, SAPI via PowerShell on Windows, `espeak` on Linux) — reliably audible but flat, robotic, and inexpressive. For a voice assistant that speaks every turn, the difference between a system voice and a modern neural voice is felt daily. The user specifically wants expressiveness — the ability for Ifa to sound calm when giving a reminder, excited when delivering good news, urgent when time-sensitive. The existing backend has no mechanism for this at all.

The user runs Ifa on two machines — an NVIDIA RTX 4060 Ti (16GB VRAM) PC and an Apple M4 Pro Mac. Both have enough compute for modern neural TTS. The rest of Ifa's stack runs locally (Ollama + Mistral for LLM, faster-whisper for STT), so the offline-first posture is non-negotiable — this rules out cloud APIs (qwen3-tts-flash, ElevenLabs, OpenAI TTS).

This brainstorm defines the upgrade: **Parler-TTS as the primary backend, with the native backend retained as a fallback**. Parler was chosen over alternatives (Bark, XTTS-v2, Kokoro) because its natural-language style-prompt API is the best match for Ifa's LLM-driven architecture and because it runs at conversational speed (~1-3s/utterance) on both target machines.

## Requirements

### Neural backend
- R1. Parler-TTS is the primary TTS backend. It is the default path every `TTSService.speak()` call takes when the model is available.
- R2. Synthesis runs **fully offline** once the model is downloaded. No cloud API dependency in the steady-state path.
- R3. Hardware backend is auto-detected: CUDA on the NVIDIA PC, MPS on Apple Silicon, CPU as a last resort. No configuration change is required to move the repo between machines — it "just works" on either.
- R4. The model loads **lazily on first `speak()` call**, not at startup. Ifa boots instantly; first utterance of a session has a one-time warm-up delay.
- R5. After first load, the model stays resident in memory for the rest of the session.

### Style control
- R6. `brain.think()` returns structured output containing both the response text **and** a style selected from a small fixed enum (initial vocabulary: `neutral`, `calm`, `excited`, `urgent`, `whispered` — subject to refinement during planning based on what Parler reliably produces).
- R7. Once finalized (during Unit 1 prompt-engineering validation), the style enum is stable and each value maps to a fixed Parler style prompt defined in one place. Enum values may change between brainstorm and Unit 1 completion based on what Parler can reliably reproduce — but not after Unit 1 ships.
- R8. If the LLM produces malformed output or an unknown style value, the system falls back to `neutral` rather than failing. A one-line warning is logged but the turn is not dropped.
- R9. The `TTSService.speak()` signature accepts an optional `style` parameter. When omitted, it defaults to `neutral`.

### Fallback behavior
- R10. If Parler fails to load (model not yet downloaded, VRAM exhausted, GPU driver issue, dependency import error), `TTSService` falls back to the existing native backend (`say`+`afplay` on macOS, PowerShell on Windows, `espeak` on Linux). Fallback also triggers on **synthesis-time failures** (e.g., CUDA OOM mid-utterance, `RuntimeError` during `generate()`), not only on initial load.
- R11. The fallback emits a single user-visible notice (e.g., "Neural TTS unavailable; using system voice — [reason]") so the user knows which mode is active.
- R12. Fallback is **per-session sticky** — once a session has fallen back to native, it stays there until Ifa restarts. No per-utterance retry.
- R13. If native synthesis also fails (no `say`, no PowerShell, no `espeak`), the system continues text-only as it does today.

### Call site preservation
- R14. The existing `TTSService.speak(text, style=...)` interface is the single call site for all speech. Callers in `ifa/skills/manager.py` and `ifa/skills/reminder.py` don't change beyond optionally passing a style. **Exception**: `ifa/core/orchestrator.py` calls `brain.think()` and passes its return value directly to `tts.speak()` and `print()`. Because R6 changes `think()`'s return type to a structured object, the orchestrator must unpack `text` and `style` from the result before calling `speak()` and `print()`. This change is in scope for the planning unit that implements R6.
- R15. Reminders (fired from daemon threads) also use the neural backend and benefit from expressive output.

## Success Criteria

- First `/ce:work` implementation run on each machine (PC and Mac) produces audible neural-quality TTS, no config change in between.
- Voice output for an "excited" style is noticeably more animated than for "calm" — the expressiveness is real and not just marketing.
- A reminder fired mid-session uses the neural backend at the same quality as a main-loop reply.
- If the model fails to load (simulated by deleting the cache), Ifa still speaks via native fallback — no silent failure.
- Moving the repo from PC to Mac (or vice versa) requires zero edits to configuration.

## Scope Boundaries

- **Cloud TTS (qwen3-tts-flash, ElevenLabs, OpenAI, edge-tts) is not in scope.** Offline is a hard requirement.
- **Voice cloning (XTTS-v2 feature) is not in scope.** No reference-audio uploads, no persona customization beyond the style enum.
- **Manual device override is not in scope.** Auto-detection handles the two-machine case. If a future need for override arises, an env var can be added — not now.
- **Multiple concurrent neural backends (Parler + Kokoro + Bark hybrid) is not in scope.** One neural backend, one native fallback.
- **STT upgrade is not in scope.** This work is output-only.
- **Arbitrary style prompts from the LLM is not in scope.** Only a small enum. Free-form prompt-generation is deferred unless the enum turns out to be insufficient.
- **Real-time streaming audio (token-by-token playback) is not in scope.** Full-utterance generation is acceptable at Parler speeds.

## Key Decisions

- **Parler-TTS over Bark / XTTS-v2 / Kokoro**: Parler hits the expressiveness-for-speed sweet spot, has both CUDA and MPS backends via torch, and its prompt-based style API fits LLM-driven architectures naturally. Bark is too slow (5-10s/utterance) for a chat loop. XTTS-v2 has a narrower emotion vocabulary and a non-commercial license. Kokoro is fast but not expressive.
- **Structured LLM output over inline tags or heuristics**: Having `brain.think()` return `{text, style}` is testable, debuggable, and keeps style decisions explicitly in the LLM's control. Inline tag parsing (`[excited] ...`) is brittle; text-based heuristics (`!` → excited) are dumb.
- **Lazy model load over eager load or async preload**: Startup stays instant. First-utterance warm-up is a one-time cost. Async preload adds complexity for marginal gain.
- **Auto-detect device only, no manual override**: User's "toggle between machines" intent is satisfied by `torch.device` auto-detection. Adding a config knob without a specific need would be premature.
- **Native backend retained, not removed**: The subprocess-based TTS we just shipped becomes the fallback. Zero risk of regressing to silent macOS if Parler fails.
- **Preserve existing TTSService interface**: Build the neural backend behind the current `speak()` method. No ripple-through to orchestrator/skills beyond adding an optional `style` parameter.

## Dependencies / Assumptions

- Parler-TTS model weights (~900MB for `parler-tts-mini-v1.1`, ~3GB for `parler-tts-large-v1`; variant chosen in Unit 1) will download on first use via HuggingFace Hub (`huggingface_hub` is already in `ifa/requirements.txt`). Parler weights license must be verified as permissive (Apache 2.0 or equivalent) before committing to the chosen variant — the prior brainstorm disqualified XTTS-v2 for license reasons and Parler should receive the same scrutiny.
- `torch` is a new dependency (several hundred MB) and is accepted as necessary.
- User has HuggingFace access (no login required for Parler public weights) and sufficient disk space.
- Mistral (via Ollama) can be prompted to emit a structured `{text, style}` response reliably enough to make per-turn style control useful (rather than a fallback-to-neutral no-op). The current `brain.think()` invokes Ollama via `subprocess.run(["ollama", "run", ...])` which does **not** natively enforce JSON output. This assumption therefore depends on either (a) switching to Ollama's HTTP API with `format: "json"` + JSON Schema, (b) using the `--format json` CLI flag, or (c) robust post-hoc parsing. The assumption is not yet validated and is listed as an open question below.
- The user's RTX 4060 Ti 16GB has comfortable headroom for Parler large. The M4 Pro's unified memory will allocate to Parler via MPS. Both assumed to handle the model without aggressive quantization.

## Outstanding Questions

### Resolve Before Planning

- [Affects R15][Needs research] **Daemon-thread safety of Parler inference on MPS.** Review reviewers disagree on whether MPS imposes a main-thread constraint comparable to pyttsx3's Cocoa NSRunLoop issue (feasibility says no, adversarial says yes-historically). Because R15 requires reminder daemon threads to invoke neural TTS on Apple Silicon — the user's primary machine — this is not safely deferrable. A 10-minute spike on the M4 Pro (load a small torch model on MPS, call `generate()` from a daemon thread, confirm no crash / silent fallback / hang) resolves this before planning commits to an architecture.
- [Affects R6, R8][Technical] **Mistral structured-output reliability via current Ollama integration.** `brain.think()` uses `subprocess.run(["ollama", "run", MODEL, prompt])` — raw CLI with no JSON mode. A short prototype should measure the proportion of Mistral turns that emit valid `{text, style}` JSON with (a) `--format json` CLI flag, or (b) HTTP API + JSON Schema. If reliability is <80% without a parser, the plan must pick one of: switch to Ollama HTTP API, add a JSON-extraction post-parser, or drop the structured-output requirement and use text-only with always-neutral style as v1.

### Deferred to Implementation

- [Affects R1, R3][Technical] Which Parler-TTS variant? `parler-tts-mini-v1.1` (~900MB, faster, somewhat less expressive) vs `parler-tts-large-v1` (~3GB, slower, richer). Decide during Unit 1 based on benchmarks on both target machines.
- [Affects R6][Needs research] Exact style enum vocabulary. The doc lists an initial set (`neutral`, `calm`, `excited`, `urgent`, `whispered`) but the final vocabulary should be validated against Parler's actual controllability — some styles may not be reliably reproducible, others may be worth adding.
- [Affects R6, R7][Technical] Exact Parler-style-prompt for each enum value. This is prompt engineering work that belongs in Unit 1's test phase — e.g., `calm` might map to "a calm, measured female voice" or something else that produces consistent output.
- [Affects R6][Technical] How to prompt Mistral to reliably emit structured `{text, style}`. Options: Ollama's native JSON mode, a strict system prompt template, or a lightweight post-processing parser with fallback. Decide in Unit 2.
- [Affects R10, R11][Technical] Exact error classes that trigger fallback. Parler/torch can raise `OSError`, `ImportError`, `RuntimeError` (CUDA OOM), `HTTPError` (model download), and more. Catalog during implementation.
- [Affects R15][Needs research] Daemon-thread safety of Parler inference on MPS. PyTorch MPS has historically had main-thread limitations on macOS; verify that `pyttsx3`-style main-thread gating isn't also needed here. If it is, reintroduce the main-thread queue we removed.
- [Affects R4][Technical] Where does the model cache live? HuggingFace default (`~/.cache/huggingface/`), or a project-local cache for reproducibility? Prefer default unless there's reason to isolate.

## Next Steps

-> Resume `/ce:brainstorm` (or run a 10-minute spike directly) to resolve the two **Resolve Before Planning** items. Once resolved, `-> /ce:plan`.
