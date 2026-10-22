# Idea: "What mistake did I make?" (gameplay review)

**Status:** future scope, not started. Captured 2026-10-22.

While playing, you ask Ifa "what mistake did I make?". It looks at the last 20–30 seconds of gameplay and answers in a sentence or two.

---

## The key constraint

Local vision models (and Ollama) take **still images, not video**. So "watching the clip" really means picking a handful of frames and asking about them as a sequence. Everything below follows from that.

## Chosen approach: reuse NVIDIA's Instant Replay buffer

NVIDIA's overlay (NVIDIA App / GeForce Experience) already keeps a rolling buffer of the last N minutes, recorded on the GPU's NVENC encoder at almost no cost. NVIDIA has no public API for reading that buffer, but Ifa can ask it to save the buffer and then read the file:

```
you: "what mistake did I make?"
Ifa ─► presses Alt+F10 (simulated) ─► NVIDIA saves the buffer as an MP4
    ─► Ifa watches the clips folder until the new file is finished
    ─► pulls ~12 frames from the last 20–30 s (PyAV)
    ─► vision model, with the game's name and your question
    ─► 1–2 sentence spoken answer
```

**Why this approach**

- No capture code, and no black frames in fullscreen games.
- No extra load while you play: NVENC is recording anyway.
- Full-quality footage, and the clip stays saved so you can watch it too.
- "Ifa, clip that" comes almost for free with the same code.

**Catches**

- Saving takes about 2–5 s before the file can be read.
- NVIDIA doesn't confirm the save. Ifa only knows it worked when a new MP4 appears; if none appears within about 10 s, Ifa says so (Instant Replay is probably off).
- Setup on the PC:
  - Turn Instant Replay on in the overlay (Alt+Z).
  - Tell Ifa the clips folder. NVIDIA usually saves to `Videos\<Game name>\` with "DVR" in the file name.
  - Tell Ifa the save hotkey if it isn't Alt+F10.
- Needs **PyAV**, a pip wheel that bundles its own ffmpeg and works offline.

**Fallback if NVIDIA doesn't work out:** Ifa keeps its own ring buffer, grabbing a small screenshot every 0.5 s for the last 60 s in RAM (about 5–10 MB), using `dxcam` (Windows Desktop Duplication). OBS's Replay Buffer over obs-websocket is another option, and it does confirm each save with the file path.

## Design notes

- **Tool as sub-agent.** A `review_gameplay(question, seconds)` tool makes its own vision call with a gameplay-coach prompt and returns only the verdict. The 12 images never enter the main conversation or its memory.
- **Sparse temporal sampling.** Spread frames evenly over the window, plus extra frames where the picture changes sharply (a death screen, a flash).
- **Frame size.** Shrink frames to about 640 px.
- **Game context.** The foreground window title names the game; put it in the prompt.
- **Builds on `analyze_screen`.** [ifa/skills/vision/vision.py](../../ifa/skills/vision/vision.py) already sends one screenshot to the main model, which accepts images.

## Limits to expect

1. **Context.** Each image is roughly 300–1,000 tokens and the context is 8k, so about 8–12 frames per question.
2. **What the 9B model can judge.**
   - It can catch positional and decision mistakes: "pushed alone into the open on low HP", "reloading when he peeked", "didn't check left".
   - It can't judge aim, crosshair placement or split-second timing from half-second stills.
   - Its knowledge of specific games is shallow.
3. **GPU sharing.** The game, the 9B model, Whisper and Chatterbox share 16 GB of VRAM.
   - Asking mid-fight may stutter the game.
   - An answer will likely take 15–40 s.
   - It works best between rounds or after a death.

## First step when we pick this up

Run a quick experiment before building anything:

1. Save a clip of a real mistake with Alt+F10.
2. Run a small script that pulls 12 frames from the clip and asks the model "what mistake did the player make?".
3. Decide from the answer:
   - **Useful:** build the tool and the "clip that" command.
   - **Generic ("be more careful"):** try a stronger vision model first.

**Open question:** which game(s)? Fast shooters need denser frames and different prompts than slower games.
