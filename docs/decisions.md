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
