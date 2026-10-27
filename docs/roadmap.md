# Roadmap and open items

What's planned, what's parked, what's built but unverified, and known limitations. Last updated 2026-10-23.

---

## Built but not yet verified on the PC / phone

These pass tests on the Mac but haven't run on the real hardware yet:

- [ ] **Package install on the PC:** run **Install packages** and confirm it repairs the venv (numpy 1.26.4, sympy 1.13.1). If it fails, send the "The conflict is caused by" lines.
- [ ] **Offline start:** after **Download voice models**, the console shows `[env] offline: …`, and Ifa starts with the router's internet unplugged.
- [ ] **Speaker verification:** enroll on the PC and phone, collect a few logged scores, and tune `IFA_SPEAKER_THRESHOLD`.
- [ ] **Apps tab + notes:** a note like "only Mom is urgent" changes the urgency verdict.
- [ ] **Spotify:** log in, search then play for Hindi and Punjabi songs, and confirm "open Spotify and play X" opens the app on the PC model (`qwen35:9b-VL-UC`; measured only on qwen3.5:9b).
- [ ] **run_command:** propose, then yes, runs; propose, then no, doesn't. Check the `cmd.exe` output encoding.
- [ ] **Persona / claim guard** on the PC model: short replies, no "Playing X" without a tool call.

## Next up (agreed or parked by you)

- **Stop chatter after some tool calls** (parked 2026-10-16). Plan: make the chosen tools terminal (`return_direct`), like `read_notifications`, so "Discord's open, Sir." is the whole reply.
- **Reply to notifications** ("tell Joe yes") through the WhatsApp tool.
- **Benchmark the PC model against qwen3.5:9b** with the tool bench ([ifa/tests/test_tool_call_bench.py](../ifa/tests/test_tool_call_bench.py)). An "uncensored" fine-tune may call tools less reliably.

## Ideas worth doing

- **Memory budget by tokens, not turns:** keep as many whole turns as fit (say 1,500 tokens), and summarise older ones instead of dropping them.
- **Set Ollama's `num_ctx` explicitly** (e.g. 8192). The Mac's MLX backend ignored the 4,096 default while the PC's backend enforces it, so the same conversation can behave differently on the two machines.
- **Trim tool descriptions.** They're about 2,000 tokens of every prompt, and each turn processes them before the first word.
- **Prefer this computer as the Spotify device** right after `open_app`.
- **Saved commands** that run without asking ("flush dns"), next to ask-first commands.
- **Check the wake-word audio against your voice too,** so a stranger's false wake doesn't even play the chime.
- **Multi-speaker:** recognise family members by name with limited permissions.
- **Phone off home Wi-Fi:** Tailscale, plus TLS for the socket.
- **Opus audio** between the phone and PC.
- **Gameplay review** ("what mistake did I make?"): save NVIDIA's Instant Replay buffer, pull frames, ask the vision model. See [brainstorms/2026-10-22-gameplay-review-idea.md](brainstorms/2026-10-22-gameplay-review-idea.md).

## Found while documenting the flows (2026-10-23)

Read-only review while writing [flows/](flows/README.md). Not yet fixed; this is the polish list.

**Can crash or go silent**
- [ ] `whats_app` returns pywhatkit's result (probably `None`) and `switch_mode` can return an exception object; `tool_step` calls `.strip()` on it, which can crash the PC turn loop.
- [ ] The "I couldn't figure out how to do that." give-up paths (fabrication guard twice, malformed tool calls) are never spoken or saved to memory.
- [ ] The activation API (port 8787) has no try/except: a port clash kills startup. The phone server already handles this.
- [ ] Phone disconnects mid-reply: the rest of the reply plays on the PC speakers instead of being dropped.
- [ ] Phone `cancel` doesn't stop the agent or the TTS while Ifa is thinking or speaking.

**Wrong behaviour**
- [ ] The phone always sends `priority: "normal"`, so `APP_PRIORITY` (WhatsApp = high) and "skip low priority" never apply to phone notifications.
- [ ] Dedupe is skipped when an app has a note, so repeated texts are counted (and saved to history) twice.
- [ ] `take_unread` moves bursts that are still settling to "older", and they never get an urgency check.
- [ ] `_expect_reply` never times out: after an unanswered follow-up, the next wake-word turn misses the short-command voice pass.
- [ ] All guards share one retry per turn, so a second guard in the same turn can't retry.
- [ ] The model-error retry only fires when the error mentions "tool".
- [ ] `listenSec` from the phone doesn't reach the PC's `VoiceInput`.
- [ ] Saving memories in the launcher deletes any memory the agent added while the page was open (`replace_all`).

**Logs, speed, rough edges**
- [ ] A stray `print(fallback_from)` prints `None` at every voice start, and the `[voice]` line says `whisper=small.en` while the real default is the Apex model.
- [ ] Silero VAD is loaded again on every capture, and silently falls back to an energy estimate on failure.
- [ ] Whisper loads on the first PC voice turn (only a phone connect warms it up), so the first turn is slow.
- [ ] The wake-word fallback name `"ifa"` isn't an openWakeWord built-in, so a missing `ifa.onnx` likely drops Ifa into text mode.
- [ ] `run.bat` skips the voice-model download that `run.command` and `run.ps1` do.
- [ ] `ifa.db` is relative to the working directory.
- [ ] `call_n8n_workflow` is offered to the model even with no workflows configured.
- [ ] `end_conversation`, `run_command` and `restart_ifa` get the generic filler ("One sec.").
- [ ] Fillers on the phone are spoken but don't appear in the phone chat.
- [ ] No launcher rows for `IFA_NOTIF_LISTEN_SEC`, `IFA_PHONE_HOST` or the `IFA_NOTIF_*` inbox settings. The app hard-codes discovery port 8788.
- [ ] No Hugging Face connection check in the launcher (it was lost in the 2026-10-21 reset).
- [ ] "Download voice models" has no done-check, so "Run missing steps" always re-runs it.

## Known limitations

- No barge-in (you can't interrupt Ifa while it speaks).
- Short replies (under about 1 s) are trusted for 30 s after a verified utterance rather than checked.
- Spotify, YouTube and web search, WhatsApp sending and n8n need the internet.
- The PC wasn't reachable from the Mac over the LAN while Cloudflare WARP was on (Atomicwork-managed). Pausing it wasn't enough in testing; quitting it or using Tailscale should fix that.

## Housekeeping

- Two tests fail on the Mac and predate this work. `test_wake_word` (2) asserts download behaviour that changed. `test_stt` is out of date after the Apex model change.
- The GitHub repo moved to `https://github.com/Jastagar/ifa.git`. Pushes work through the redirect; update the remote URL when convenient.
- Rotate the Spotify client secret that was once committed in history.
