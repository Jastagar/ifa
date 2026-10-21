# Only your voice (speaker verification)

After you enroll, Ifa ignores every voice that isn't yours: TV, music, family, guests. Nobody else can trigger a tool or say "yes" to a command.

**Files:** [ifa/voice/speaker.py](../../ifa/voice/speaker.py), [ifa/voice/enroll.py](../../ifa/voice/enroll.py), [ifa/voice/input.py](../../ifa/voice/input.py), [ifa/services/phone_server.py](../../ifa/services/phone_server.py); app: `YourVoice()` in `MainActivity.kt`, enrollment in `IfaVoiceService.kt`

---

## How it works

```
captured utterance ─► voiced seconds < 1.0?
                        ├─ yes ─► your voice matched on this mic in the last 30 s? accept : ignore
                        └─ no  ─► embedding (CAM++ model, ~20 ms) ─► cosine vs your voiceprint per mic
                                   ─► score ≥ 0.5? accept (and refresh trust) : ignore, never transcribed
```

- A **speaker embedding model** turns a few seconds of speech into a 512-number voiceprint that's similar for the same person and different for others, whatever the words.
- **Enrollment:** you read 6 short phrases (English and Hinglish), and the average of their embeddings is your voiceprint. A separate voiceprint is kept **per mic** (`pc`, `phone`), since the two mics sound different, and each utterance is scored against the best match.
- **The check runs on the PC, after capture and before speech-to-text,** so it covers both mics. Rejected audio is never transcribed.
- The console logs every score:
  - `[voice] your voice (score 0.82, needs 0.50)`
  - `[voice] ignored: not your voice (score 0.23, needs 0.50)`

## Decisions and why

- **What happens to other voices: ignore them completely** (you chose this). The alternatives were "chat but no actions", which was the April plan, and "recognise several people".
- **Enroll on both mics** (you chose this): from the launcher (PC mic, `python -m ifa.voice.enroll`) and from the app (Settings → Voice → Teach Ifa my voice).
- **Runtime: sherpa-onnx.** It bundles the audio feature extraction these models expect, has wheels for Windows and Apple Silicon, and adds no torch dependency.
- **Model: 3D-Speaker CAM++ zh_en "advanced"** (28 MB). Three candidate models were tested on real recordings of 7 speakers (sherpa-onnx's test clips, including a speaker in both Chinese and English):

  | Model | Lowest same-person score | Highest stranger score | Separates everyone? |
  |---|---|---|---|
  | WeSpeaker CAM++ (VoxCeleb) | 0.16 | 0.91 | no |
  | WeSpeaker ResNet34 (VoxCeleb) | 0.80 | 0.88 | no |
  | **3D-Speaker CAM++ zh_en** | **0.60** | **0.46** | **yes** |

  With a 3-clip voiceprint, the enrolled person's new clips scored **0.80–0.88**, and the closest stranger scored **0.42**. That's why the threshold is **0.5**.
  - **Lesson:** synthetic macOS voices were useless for this test. They all come from one engine, so the model scored them as nearly the same person. Use real recordings.
- **Short replies can't be verified.** A 0.8 s clip of the right speaker scored only 0.42. So anything under about 1 s of speech ("yes", "haan") counts as you only if your voice was matched on the same mic in the last `IFA_SPEAKER_TRUST_SEC` (30 s). That's what lets you confirm a `run_command` after Ifa asks. A stranger's voice never extends the trust window.
- **Enrollment quality check:**
  - each recording needs ≥ 1.2 s of speech
  - at least 3 good recordings are needed
  - a recording that doesn't sound like the others (similarity < 0.4 to the rest) is left out
  - if they don't sound like one person, nothing is saved
- **Privacy:** `voiceprint.json` holds embeddings only (no audio), is readable only by you (mode 600) and is gitignored. It records which model made it, and a voiceprint from another model is ignored with a message to enroll again.
- **Failing safe:** if sherpa-onnx or the model is missing, the check is **off** (Ifa listens to everyone) and the startup line says why. It never locks you out.

## Settings

`IFA_SPEAKER_CHECK` (on), `IFA_SPEAKER_THRESHOLD` (0.5), `IFA_SPEAKER_TRUST_SEC` (30), all in the launcher under Listening. Lower the threshold if Ifa ignores you; raise it if it answers someone else. Use the logged scores to choose.

## Limits and next

- The check runs after capture, so a false wake word still plays the chime. Possible fix: also check the wake-word audio.
- Not yet tuned on your real voice and mics. Collect a few logged scores from the PC and phone, then set the threshold.
- Possible next step: recognise several people by name (multi-speaker), so Ifa can greet them or limit what they can do.
