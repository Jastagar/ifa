# The Android app and its connection to Ifa

The phone is a remote mic, speaker and notification source for Ifa on the PC. Most of this was built in other sessions (`ad4197e`…`b7a732d` here, `31c6b19`…`f5752ee` in the app). Notification settings, enrollment and the Apps tab were added on 2026-10-21.

**Repos:**
- Ifa (PC): [ifa/services/phone_server.py](../../ifa/services/phone_server.py), [ifa/services/pairing.py](../../ifa/services/pairing.py)
- App: `~/Projects/ifa.app` (Kotlin + Compose, branch `master`). Key files: `service/IfaVoiceService.kt`, `ifa/IfaConnection.kt`, `ifa/IfaDiscovery.kt`, `voice/WakeWordEngine.kt`, `AppsScreen.kt`

---

## Shape

```
 phone                                              PC (Ifa)
 ─────                                              ────────
 wake word "Ifa" on the phone (openWakeWord port, same ifa.onnx)
 mic ─► 16 kHz PCM16, 80 ms frames ──── WebSocket ────► VAD ─► voice check ─► Whisper ─► agent turn
 speaker ◄── Chatterbox audio frames ◄──────────────── TTS (streams to the phone instead of PC speakers)
 notifications, settings, enrollment ─────────────────► NotificationCenter, SpeakerCheck
```

**Decisions and why**
- **One WebSocket, straight to Python**, with JSON text frames for events and binary frames for audio. The old Spring Boot bridge is skipped, since an extra hop adds latency and something else to keep running. It's still in the app behind "Also send to the backend URL", off by default.
- **The wake word runs on the phone.** No audio leaves the phone until it hears "Ifa", which is better for battery, bandwidth and privacy than streaming continuously. The phone's port matches the Python scores exactly.
- **Everything after capture runs on the PC** (VAD, the voice check, Whisper, the agent, TTS), so there's one brain and one memory. Phone and PC turns go through one queue and never overlap.
- **The phone takes over when connected** (`IFA_PHONE_PRIORITY`): the PC mic stops listening for the wake word, and notifications nudge the phone instead of the PC.
- **Discovery:** the app broadcasts a UDP probe on 8788 and the PC answers. If the router blocks broadcasts, type the PC's address.
- **Pairing:** a 6-digit code in the `X-IFA-Token` header. It's made once, saved in `.phone_token`, and shown in the launcher's Phone app row. Repeated wrong codes get a lockout (HTTP 429), and the app waits a minute before retrying.
- **Phone mics record quietly** (peaks around 0.01 in voice-recognition mode). The app applies a learned per-utterance gain (1–16×), and the PC boosts quiet audio (up to 20×) before Whisper.
- **Speech settings travel with each utterance** (`silenceMs`, `vadThreshold`), so the phone's pause length doesn't change the PC mic's.

## Protocol

`ws://<pc>:8789/phone`, header `X-IFA-Token: <pairing code>`.

| Phone → PC | Meaning |
|---|---|
| `start {startTimeoutMs, silenceMs, vadThreshold, enroll?}` + binary PCM16 frames | Start capturing an utterance (`enroll: true` keeps it for the voiceprint instead of transcribing) |
| `stop {reason}` / `cancel` | The phone ended the utterance (silence or the button) / abandon it |
| `text {text}` | A typed message |
| `notification {packageName, appName, title, text, priority, appNote}` | A phone notification ([notifications.md](notifications.md)) |
| `notification_settings {settleSec, ttlSec, maxRecent, maxDiscarded, listenSec}` | Sent on every connect and on change |
| `enroll_begin` / `enroll_finish` / `enroll_cancel` / `enroll_forget` | Voice enrollment ([speaker-verification.md](speaker-verification.md)) |
| `ping` | Keep-alive |

| PC → phone | Meaning |
|---|---|
| `ready {version, name, ttsSampleRate, wakeThreshold, wakeConsecutive, followupSec, voiceSources}` | Connected; the PC's wake settings and which mics have an enrolled voice |
| `listening`, `end_of_speech`, `transcript {text}` | Capture progress |
| `no_speech {reason}` | Nothing usable: too quiet, no words, **or not your voice** |
| `reply {text}`, `audio {sampleRate}` + binary frames | Ifa's answer and voice |
| `turn_end {followup, followupSec, cue}` | Turn over; whether to listen again without the wake word |
| `nudge {level}` | A notification arrived (`urgent` wakes the screen and plays the announcement first) |
| `notification_settings {...}` | The values the PC applied |
| `enroll_ready {phrases}`, `enroll_sample {ok, count, seconds, message}`, `enroll_done {ok, message, sources}` | Enrollment steps |
| `error {message}`, `pong` | Errors (`busy` while a turn is running) and keep-alive |

Unknown message types get an `error` reply, so the app and PC can be updated at different times without crashing.

## App screens

- **Ifa tab:** status, mic button, wake-score meter, chat, text box, Settings (connection, voice, "Your voice", notification timing), Logs.
- **Apps tab:** installed apps with a switch and a note each ([notifications.md](notifications.md)).
- **Background:** a foreground service keeps the wake word running with the screen off. After a phone restart the app has to be opened once, because Android doesn't let an app turn on the mic from the background at boot.

## Building

The app's `local.properties` points at the Windows SDK, because it's built and installed from Android Studio on the PC. To compile-check on the Mac, copy the repo to a scratch folder with its own `local.properties` and run Homebrew `gradle` with JDK 21 (`:app:compileDebugKotlin`).

## Next

- Opus compression instead of raw PCM. Not needed on home Wi-Fi.
- Use away from home (Tailscale) and a token-protected TLS endpoint.
- Choosing which device answers when both the PC and the phone hear you.
