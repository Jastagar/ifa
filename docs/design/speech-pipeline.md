# Speech pipeline and latency

From your voice to Ifa's voice, and the work on making the first sound come fast.

**Files:** [ifa/voice/input.py](../../ifa/voice/input.py), [ifa/voice/wake_word.py](../../ifa/voice/wake_word.py), [ifa/voice/capture.py](../../ifa/voice/capture.py), [ifa/voice/stt.py](../../ifa/voice/stt.py), [ifa/services/tts_service.py](../../ifa/services/tts_service.py), [ifa/services/fillers.py](../../ifa/services/fillers.py), [ifa/core/orchestrator.py](../../ifa/core/orchestrator.py) (`run_turn`), [ifa/utils/latency.py](../../ifa/utils/latency.py)

---

## Pipeline

```
mic 16 kHz ─► wake word (openWakeWord, ifa.onnx) ─► chime ─► capture (Silero VAD, ends on silence)
          ─► voice check (speaker-verification.md) ─► Whisper (Oriserve Hinglish "Apex")
          ─► agent turn (agent-loop.md) ─► sentences ─► Chatterbox Turbo ─► audio queue ─► speaker
                                          └─ filler clip if the answer is slow
after a reply: listen for IFA_FOLLOWUP_WINDOW_SEC without the wake word
```

The phone app runs the same pipeline with its own mic and speaker. Its wake word runs on the phone; everything after capture runs on the PC ([phone-app.md](phone-app.md)).

## Measuring first: TTFS

**Time to first speech (TTFS)** is the time from you finishing speaking to the first audio of the real answer. It's the number that decides how responsive Ifa feels.

A `TurnTimer` follows each turn across the voice, agent and TTS threads and prints a breakdown:
- VAD wait and speech-to-text time
- for every model call: time to first token, prefill tokens and time, decode speed, reloads
- filler and first-sentence timing, and TTS speed
- when audio actually reached the speaker

`IFA_LATENCY_LOG=path.jsonl` appends every turn as JSON. **Pattern: measure before optimising.** Every change below was justified by these numbers.

Measured on an M4 Pro with qwen3.5:9b:
- the prompt is about 2,000 tokens, about 1,700 of them tool schemas
- prefill is the main cost of the first token, and Ollama's prompt cache helps when the prompt start is unchanged

## Decision: stream the model

`stream_chat` used to send `"stream": false`, so the first sentence waited for the last token. With streaming, sentences go to TTS while the rest is still generating. **Saved 1.3–2 s** before the first sentence on a typical answer.

Streaming exposed a race: an acknowledgement could play *after* the answer had started. Checking whether the answer has started and queueing the filler now happen as one step under a lock.

## Decision: one model, no "ack model"

**The idea:** a tiny second model (qwen3.5:0.8b) would say "Okay, on it" quickly while the 9B worked.

**Why it was dropped:**
- The ack still took **1.3–2.4 s** to the first sound, because it had to generate and then synthesize.
- It competed with the main model for the GPU, and two Ollama instances caused model reloads.
- It made false claims ("Reminder set") before anything had happened.

**Replacement: pre-rendered filler clips.** About 15 phrases are rendered once in Ifa's voice at startup and played straight onto the audio queue, so first sound is about **0.65 s**.
- A filler plays at most once per turn: as soon as a tool is called (with that tool's phrase, e.g. "Opening that."), or after `IFA_FILLER_DELAY_MS` (650) of silence, and never after the answer has started.
- **Generic phrases must be true for any request:** "One sec.", "Just a sec.", "One moment.", "Give me a second." "Hmm, let me think" before "open Spotify" sounded wrong, because the generic filler plays before the model has picked a tool, so it can't know the intent. Override them with `IFA_FILLER_PHRASES`.

## Speech-to-text

- **Model:** Oriserve **Whisper-Hindi2Hinglish-Apex** (large-v3-turbo, 0.8B), chosen in another session (`916b4c1`) for much lower error on conversational Hinglish than "Swift" (WER 47.6 vs 65.2). It runs best on a GPU.
- **The language token is `en`, as the model card trains it.** Forcing `hi` spelled English words in Hindi. Set with `IFA_WHISPER_LANGUAGE`; `IFA_WHISPER_PROMPT` can bias it toward names.
- **Device:** `IFA_WHISPER_DEVICE` (`auto`/`cuda`/`cpu`). `auto` tries CUDA and falls back to CPU.

## Text-to-speech

- **Chatterbox Turbo** with a cloned voice (`IFA_VOICE_SAMPLE`) on `IFA_TTS_DEVICE`.
- Sentences are synthesized in a producer thread and played by a consumer thread with a bounded queue, so synthesis of sentence 2 overlaps playback of sentence 1.
- `is_speaking` mutes the mic while Ifa talks (plus a cooldown), so it doesn't hear itself.

## Known limits

- No barge-in: you can't interrupt Ifa mid-sentence. This was deferred because replies are short.
- A false wake word still plays the chime before the voice check ignores the audio.
- On Apple Silicon, Whisper (fp16) and Chatterbox run on the Apple GPU (`mps`). Sentence splitting is plain Python ([ifa/utils/sentences.py](../../ifa/utils/sentences.py)); it replaced `blingfire`, which only shipped an Intel build for macOS.
