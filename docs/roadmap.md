# Roadmap and open items

What's planned, what's parked, what's built but unverified, and known limitations. Last updated 2026-10-21.

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

## Known limitations

- No barge-in (you can't interrupt Ifa while it speaks).
- The full voice stack doesn't run on Apple Silicon: `blingfire` is Intel-only on macOS, and there's no `mps` device option.
- Short replies (under about 1 s) are trusted for 30 s after a verified utterance rather than checked.
- Spotify, YouTube and web search, WhatsApp sending and n8n need the internet.
- The PC wasn't reachable from the Mac over the LAN while Cloudflare WARP was on (Atomicwork-managed). Pausing it wasn't enough in testing; quitting it or using Tailscale should fix that.

## Housekeeping

- Three tests fail on the Mac and predate this work. `test_wake_word` (2) asserts download behaviour that changed. `test_tts_phrase_batch` (1) depends on `blingfire`, which can't load on the Mac. `test_stt` is out of date after the Apex model change.
- The GitHub repo moved to `https://github.com/Jastagar/ifa.git`. Pushes work through the redirect; update the remote URL when convenient.
- Rotate the Spotify client secret that was once committed in history.
