# Decision log

Every significant choice since the October 2026 rework: what we decided, why, and what we rejected. Newest last. The design documents in [design/](design/) explain each area in depth. Commits are in this repo unless marked "app" (`~/Projects/ifa.app`).

---

### 1. Remove dead code from earlier architectures (`67392b3`, 2026-10-15)
- **Decision:** delete the keyword-routed Skill classes, vision intent routing, ActivationService, the plan-and-execute draft, the facts table, `*_bkp` files and empty modules. Stop tracking `.cache` and `PyWhatKit_DB.txt`.
- **Why:** each design change (keyword routing → LLM tools, SQLite facts → memory file, HTTP turns → wake-up) left code nothing called. Dead code misleads readers and tests.

### 2. ReAct loop with a 3-hop budget (`170a9df`)
- **Decision:** let the model act, observe and act again, up to 3 tool rounds, then answer without tools. Every call gets a result, errors included.
- **Why:** one tool per turn couldn't handle "look at my screen and remind me". Stopping at the first error broke the call/result protocol.
- **Rejected:** unbounded loops, which can spin forever on a confused model.

### 3. Measure time to first speech before optimising (`ca61462`)
- **Decision:** a `TurnTimer` across threads and a JSONL log.
- **Why:** guessing where latency went was wrong more than once. The numbers drove every decision after this.

### 4. Stream the model (`4d37a4b`)
- **Why:** with streaming off, the first sentence waited for the last token. Streaming saves 1.3–2 s.
- **Follow-up:** checking whether the answer started and queueing the filler became one step under a lock, which fixed a race where the ack played after the answer.

### 5. One model; pre-rendered fillers instead of an "ack model" (`f2bd967`, `428f4f3`)
- **Decision:** remove the 0.8B acknowledgement model and its second Ollama instance. Play fixed phrases rendered once in Ifa's voice.
- **Why:**
  - First sound went from 1.3–2.4 s to about 0.65 s.
  - No GPU contention or model reloads.
  - No false claims like "Reminder set" before anything happened.
- **Rejected:** keeping the small model on a separate instance, which caused reloads and used too much memory.

### 6. Notifications as an event inbox with bursts, debounce and two queues (`d76e006`)
- **Why:** people split one thought across messages; announcements must not interrupt a conversation; nothing should be lost, but nothing should pile up.
- **Your choices:** the LLM decides urgency; mid-chat notifications wait until after; recent queue of 7 for 90 s, then a discarded queue read only on request.

### 7. "What is it?" as a terminal tool with a focused summarizer (`d76e006`)
- **Why:** the main model wrote summaries from the wrong point of view and offered to reply. A narrow prompt plus speaking the result as-is fixed both.
- **Lesson:** a persona rule "never reply to notifications" made the model skip the tool entirely. Negative instructions spread to cases they weren't meant for.

### 8. Guard against fabricated tool results (`d76e006`)
- **Why:** qwen3.5:9b once wrote a fake `TOOL_RESULT` block. A code check is more reliable than another prompt rule.

### 9. Remove comments and docstrings from `ifa/` (`0ba887e`)
- **Your call.** It was verified with the parser so that no code or prompt text changed.

### 10. A localhost launcher driven by one settings list (`e955d67`, `3ecf01c`)
- **Why:** you wanted keys, GPU/CPU choices and environment variables in a UI, with installation and setup there too.
- **Security:** it binds to localhost, needs a session token and a local Host header, and never echoes secrets. Setup commands are an allowlist of named steps.

### 11. Generic fillers must not assume what you asked (`ace1e88`)
- **Why:** they play before a tool is chosen. "Let me think" before "open Spotify" was wrong; "One sec." is true for any request.

### 12. Spotify login as its own step, never inside a turn (`ee2e943`)
- **Why:** untracking `.cache` deleted the PC's token on pull. A browser login can't complete in the middle of a voice turn.

### 13. Tool descriptions say how tools combine (`d69ab63`)
- **Why:** "open Spotify and play X" sometimes only played. The `open_app` description also told the model to ask for confirmation first. Measured 15/16 → 20/20.

### 14. run_command: any command, confirmed in code; apps as a name enum (`38cf22e`)
- **Your choice:** any command, ask first.
- **Why code, not prompt:** prompt rules are suggestions and the model reads third-party text. A destructive-pattern deny list backs it up.
- **Apps:** the model picks a name and the command comes from `apps.json`. They're folded into `open_app`, not a second tool.

### 15. Persona: one or two sentences, examples over rules (`04a8637`)
- **Why:** the persona asked for "4–5 sentences", which produced paragraphs and closing offers. Measured 2–5 → 1–3 sentences, with Hinglish answered in English 11/12 times once an example reply was added.

### 16. Spotify: search, then play by number (`2dead3b`)
- **Your idea.** Code ranks the results (original versions first, the account's country, artist as a filter) and the model picks one.
- **Why:** taking the first result often played slowed/reverb or cover uploads of Hindi and Punjabi songs.

### 17. Memory keeps whole turns, including tool calls; action-claim guard (`603947f`)
- **Why:** text-only history taught the model to answer "play X" with "Playing X" and no tool call. The history is a few-shot prompt.

### 18. Phone as a remote mic and speaker over one WebSocket, wake word on the phone (`ad4197e`, app `31c6b19`, other session)
- **Why:** reuse the PC pipeline, keep one memory, no audio leaves the phone before the wake word, no Spring Boot hop.

### 19. Phone takes over when connected; editable prompts; stricter urgency (`b7a732d`, other session)

### 20. End on "that's all"; restart through a supervisor (`ac96f8a`, other session)

### 21. Memory by topic in SQLite (`99ed574`, other session)
- **Why:** `memory.md` only grew and went whole into every prompt.

### 22. Apex Hinglish Whisper with the `en` token (`916b4c1`, other session)
- **Why:** much lower error on conversational Hinglish. Forcing `hi` spelled English in Hindi.

### 23. Notification settings move to the phone; per-app switch and note (`838e34a`, `7156531`, app `ada9e51`, `341b212`)
- **Your call:** settings in the app, not the web UI; a list of installed apps with a toggle and a context note each.
- **Why the note goes outside `<notification>`:** your note is trusted and the message isn't, so a message can't pretend to be your instruction.
- **Rejected:** High/Normal/Low chips (my first version). The switch and note express what you meant.

### 24. Speaker verification: ignore other voices; enroll on both mics; CAM++ model (`55aaef1`, app `8cf5667`)
- **Your choices:** ignore other voices completely; enroll from both the PC and the phone.
- **Model:** the only one of three that separated every speaker in real recordings. Threshold 0.5 from measured margins (own 0.80–0.88, strangers ≤ 0.42).
- **Short replies** are trusted only right after a matched utterance.

### 25. Offline mode decided at startup from the model cache (`3007cfc`)
- **Why:** Hub checks crashed Whisper loading offline, `run.bat` forced online, and pywhatkit pinged google.com on import. Verified with a network guard: 0 attempts.
- **Rejected:** always forcing offline, which would break the first run before the models are downloaded.

### 26. One package lock, Python 3.10–3.12, one install script (`9c86238`, `d361209`)
- **Why:** separate install steps silently overwrote each other's versions (numpy 2, sympy 1.14), which broke torch and transformers. The first lock only worked on 3.12.
- **Rule:** versions change only in `constraints.txt`.

### 27. Pin setuptools to 81; check imports after installing; never crash on the watermark
- **Why:** a package that needs another package without naming a version still gets whatever is newest. setuptools 82+ removed `pkg_resources`, perth hid the resulting import error, and Chatterbox failed with `'NoneType' object is not callable`.
- **What changed:**
  - setuptools is pinned to 81.0.0 and listed in `requirements.txt`, so it's always covered by the lock.
  - `install_packages` tries importing torch, transformers, Chatterbox (and its watermarker), sherpa-onnx, openwakeword and Whisper after installing.
  - `tts_service` falls back to perth's no-watermark class. The watermark only marks the audio as AI-generated, which a personal assistant doesn't need.

### 28. Read notifications only when asked, enforced in code
- **Why:** the unread note on every message, plus a turn-ending tool, plus memory of earlier calls, made Ifa read notifications instead of answering. A stronger prompt wouldn't be reliable on a 9B model.
- **What changed:** the note and the tool apply only right after an announcement (90 s) or when your message mentions notifications or messages. Otherwise the tool refuses with an error, and the turn continues to answer you.

### 29. Act on clear requests; catch promises as well as claims
- **Why:** you saw Ifa re-confirming clear requests on the PC. On qwen3.5:9b it asked back 0/12 times, so the likely cause is the PC model applying run_command's "ask first" too broadly, or an edited persona. The probe also showed two other problems:
  - the model parroting the persona's example ("Discord's open, Sir.") with no tool call
  - promising ("Let me check…", "I'll send…") with no tool call
- **What changed:**
  - The persona says to act right away on clear requests and to ask only when something is missing. run_command is named as the only tool that always asks.
  - The example sentence was removed.
  - The action check also catches a first sentence that promises an action when no tool has run.
- **Measured:** 18 requests, 1 ask-back (a reasonable WhatsApp detail question). run_command still proposes first.

### 30. Google Calendar via REST and official auth libraries; dates in code; Quick Add
- **Why:** you asked for a calendar skill.
- **Library:** `google-api-python-client` would pull protobuf, which conflicts with the lock. The REST API plus `google-auth` fit without changing a pin.
- **Dates:** small models get dates wrong, so `list` ranges are computed in code and `add` relies on Google's Quick Add parser, with Ifa repeating back what Google understood.
- **Removal:** happens only on a single match, and lands in the trash.

### 31. Ask Ollama for an 8k context on every call
- **Why:** the PC's Ollama keeps its default 4,096-token context, and Ifa's prompt (mostly tool descriptions) is about 3,800–4,050 tokens before any conversation. When a request didn't fit, Ollama dropped the oldest messages, including your question in the middle of a tool call. Qwen 3.5's chat template then failed with "No user query found in messages", and Spotify's search-then-play steps hit it most.
- **Fix:** `IFA_OLLAMA_NUM_CTX` (8192, launcher: Language model) is sent on every request, streaming and one-off alike, so Ollama never reloads the model over a size change.
- **Lesson:** the context limit flagged in the 2026-10-16 discussion (MLX ignores it, the PC enforces it) became a real failure once the tool list grew.


### 32. Stay on the 9B dense model; no MoE for now (2026-10-27)
- **Considered:** Qwen3.6-35B-A3B (uncensored HauhauCS build, ~19 GB at IQ4_XS, vision included) and Qwen3-30B-A3B, run with ~8 GB on the GPU and the rest in the PC's 32 GB of RAM.
- **Why not:** it doesn't fit next to Chatterbox and Whisper in 16 GB of VRAM. The spill to RAM would cost about half the writing speed (est. 20–35 tok/s vs 42–52), slower prompt reading, 15–30 s reloads, and most of the PC's RAM and GPU, so no gaming or heavy multitasking while Ifa runs. Not worth it for a voice assistant.
- **Cheaper wins instead:** trim the tool descriptions (shorter prompt), keep the model loaded, and consider Q5_K_M/Q6_K of the current model if quality needs a nudge.
- **Revisit when:** a GPU with 24 GB+ of VRAM, or an MoE that fits fully in ~10 GB with tool calling as good as the 9B.
