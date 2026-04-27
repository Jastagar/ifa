---
title: "feat: Stage 3 — custom `ifa` wake-word model"
type: feat
status: active
date: 2026-04-27
origin: docs/plans/2026-04-23-003-feat-stage2-voice-input-plan.md
---

# feat: Stage 3 — custom `ifa` wake-word model

## Overview

Train an openWakeWord model that detects the single word `"ifa"`, ship the resulting ONNX file inside the repo, and make it the default `IFA_WAKE_MODEL`. This replaces the placeholder `hey_mycroft` built-in we shipped in Stage 2 — which works but is the wrong name and uses the prefixed `hey_` form the user has consistently disliked.

The original draft of this plan targeted `"hey ifa"` as a compromise, citing single-word false-positive risk. Adversarial review flagged that rationale as an *unverified phoneme claim* — and the user explicitly prefers the bare name. Pivoting to single-word `ifa`. The `hey ifa` form becomes the documented fallback if the single-word model can't hit the FP target after preemptive threshold tightening.

The runtime integration is already in place from Stage 2:

- `IFA_WAKE_MODEL` accepts a built-in name OR a filesystem path to a `.onnx`
- `WakeWordListener._derive_score_key` returns the basename-without-extension as the score-key, so `ifa.onnx` automatically maps to score key `ifa`
- Threshold + consecutive-frames guards (`IFA_WAKE_THRESHOLD`, `IFA_WAKE_CONSECUTIVE`) tune in via `.env`

So nearly all of this plan is the **training pipeline + verification + integration**, not new application code.

## Problem Frame

The user has consistently disliked the `hey_` prefix on the wake word. Stage 2 shipped `hey_mycroft` under protest as a placeholder ("functional > thematic until custom training lands"). Project memory `project_wake_word.md` flags this as still-active discomfort. An earlier draft of this plan compromised at `"hey ifa"` (better name, kept the disliked prefix); the user vetoed that compromise and asked to go straight to single-word `"ifa"`.

The path forward: train a small custom model on `"ifa"` via openWakeWord's pipeline, commit the resulting ONNX (~1 MB), and switch the default. The headline risk is that single-word detection has higher false-positive rates than two-word phrases — the plan addresses that with preemptive threshold tightening, an early FP-rate gate in Unit 3, and a documented retreat path to `hey ifa` (Stage 3.5) if the single-word model can't meet R2.

## Requirements Trace

- **R1.** Detect the spoken word `"ifa"` reliably on the repo owner's voice — target ≥ 9/10 hit rate at conversational distance over an n=10 sample, **non-regression** against `hey_mycroft` (which scores 10/10 today). The harder bar — n=10 with 9/10 floor — exists because the n=5 "≥4/5" gate the earlier draft used has no statistical power and could pass a model that's a strict regression in everyday feel.
- **R2.** False-positive rate at conversational volume **≤ 1 unintended wake per 30 minutes** AND ≤ what `hey_mycroft` produces today. Both bars matter: an absolute floor (so the system is usable) plus a non-regression check (so the swap doesn't trade naming for usability). The hey_mycroft baseline must be measured before declaring R2 met — see Unit 3.
- **R3.** Plug-in path is zero-code: drop the `.onnx` somewhere accessible, point `IFA_WAKE_MODEL` at it. (Already supported.)
- **R4.** The `.onnx` ships **inside the repo** (not in `~/.cache` or HuggingFace Hub) so a fresh clone + venv install + launcher run gives the user `ifa` immediately, with no extra setup beyond Stage 2's normal one-time download flow. File size ≤ 5 MB so this doesn't bloat the repo.
- **R5.** No additions to `ifa/requirements.txt` for *runtime* deps. Training-time deps (`torch`, `torchaudio`, training datasets) live OUTSIDE the runtime venv — likely in a Google Colab notebook session or a separate local environment.

## Scope Boundaries

- **Training is a one-time activity, not an ongoing pipeline.** No CI integration, no auto-retraining, no in-app training. The deliverable is a single committed `.onnx` artifact.
- **No multi-user voice generalization required.** Train + verify against the repo owner's voice only. Mitigation: openWakeWord's recommended synthetic-data pipeline already produces voice-diverse positives via Piper TTS — if the model fits one voice well, that's sufficient for v1.
- **No automatic *runtime* fallback that masks training failure.** If `ifa` underperforms in everyday use, the user manually flips `IFA_WAKE_MODEL` back to `hey_mycroft` (or to `hey ifa` once Stage 3.5 lands). The only automatic fallback is the **missing-file** path (covered in Unit 4) — and even that logs a prominent warning so the divergence is observable, not silent. Distinct from the "underperforms" failure mode.
- **No retraining on real recorded data in v1.** v1 stays purely synthetic — generated TTS samples + canonical negative datasets. **However, if Unit 1's first synthetic-only training run yields validation accuracy < 0.95 OR Unit 3's hit rate < 9/10, layering ~30-50 real recordings of the user saying "ifa" becomes Unit 1's contingency path**, not deferred work. The same failure mode that killed `alexa` for this user (synthetic-trained model that doesn't fit voice) is the predictable risk here, and the fix is known — don't waste multiple Colab cycles fighting it.
- **No abandoning the single-word target prematurely.** The compromise to `"hey ifa"` is documented as a Stage 3.5 fallback only if the FP-rate gate in Unit 3 cannot be satisfied via threshold/consecutive tuning AND real-recorded fine-tuning AND the user prefers prefix > tuning over more retries.

### Deferred to Separate Tasks

- **Stage 3.5 — `"hey ifa"` retreat model** — Same training pipeline targeting two-word phrase. Trigger condition: Unit 3 / Unit 5 cannot satisfy R2 even after preemptive threshold tightening AND real-recorded fine-tuning, AND the user accepts that the prefix is the only way to ship.
- **Real-recorded fine-tuning beyond the contingency path** — Beyond the 30-50 sample contingency in Unit 1, a larger 200-500-sample real-recorded layer is a future polish if everyday hit-rate degrades over time (e.g. morning voice, sick voice, accent shift). Stage 3.6 if needed.

## Context & Research

### Relevant code and patterns

- [ifa/voice/wake_word.py](../../ifa/voice/wake_word.py) — `WakeWordListener` already supports path-based model specs via `_derive_score_key`. No changes needed for the integration path; possibly only a default-value change.
- [ifa/voice/wake_word.py](../../ifa/voice/wake_word.py):48 (`_DEFAULT_MODEL = "hey_mycroft"`) — this is what gets swapped to point at the new ONNX when training is verified.
- [.env.example](../../.env.example) — documents `IFA_WAKE_MODEL` for the user; will be updated to mention the bundled custom model as the default.
- [scripts/smoke_voice.py](../../scripts/smoke_voice.py) `record` mode — already saves a WAV and offline-scores it against any `IFA_WAKE_MODEL`. Reused as the verification harness in Unit 3.
- [scripts/setup_voice_models.py](../../scripts/setup_voice_models.py) — currently calls `download_models(model_names=[...])` for built-in names. For a path-based spec, it explicitly skips the download (paths don't have an HF-hosted equivalent). No changes needed; behavior is already correct.

### Institutional learnings

- `project_wake_word.md` (memory): three earlier built-in models tried before `hey_mycroft` worked — `hey_jarvis` (3/50 live), `alexa` (0/20 live, 0.02 offline), `hey_mycroft` (1.0 offline, 10/10 live). The "fits the voice" property cannot be predicted from training corpus size alone — small-sample empirical verification is the only reliable signal.
- Stage 2 plan, Key Technical Decisions: openWakeWord's `AudioFeatures` is a stateful rolling buffer; never duplicate `predict()` calls on the listener's model. Same constraint applies to the trained model — no test path should run inference on the listener's instance.
- `scripts/smoke_voice.py` `record` mode pattern: capture 8 s of audio, save to WAV, then offline-score chunk-by-chunk. This is the gold-standard verification harness — works for any model — and has surfaced wake-word fitness issues twice now.

### External references

- **openWakeWord training notebook (canonical)**: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
  - Generates ~2000 positive samples via Piper TTS with multiple voices/speeds
  - Pulls Common Voice + MUSAN + FMA Music as negative datasets
  - Trains a small DNN (~32k parameters)
  - Exports to ONNX
  - Recommended runtime: T4 GPU on Colab free tier, ~30-60 min total
- **openWakeWord training docs**: https://github.com/dscripka/openWakeWord#training-new-models — same content as the notebook in markdown form
- **Piper TTS**: https://github.com/rhasspy/piper — local TTS engine the notebook uses for positive-sample generation. Multiple voices supported.

## Key Technical Decisions

- **Use openWakeWord's official training notebook**, not a hand-rolled pipeline. The maintainer-published notebook is a known-good reference; rebuilding it locally is a lot of yak-shaving for no incremental value (we'd still end up with the same ONNX). This makes "training" a 1-2 hour operator activity with mostly waiting, rather than a Python project of its own.
- **Run training in Google Colab** (free tier, T4 GPU). Avoids:
  - Adding ~1 GB of training-time deps (torch + torchaudio + datasets) to local environment
  - Downloading ~10 GB of negative datasets to user's PC
  - Local-vs-Colab CUDA version mismatches we already fought once with Whisper
  - The user's PC GPU isn't strictly required; the notebook is sized for Colab T4
- **Commit the trained `.onnx` directly to the repo** at `ifa/models/ifa.onnx`. File is ~1 MB, well under GitHub's 100 MB limit, no need for LFS. Committing means a fresh clone gives the user the right wake-word with no manual setup steps.
- **Make `ifa.onnx` the new `_DEFAULT_MODEL`** in `wake_word.py` — but expressed as a path resolved relative to the package, not a built-in name. Keep `hey_mycroft` as a documented fallback option in `.env.example` so users without the trained model (e.g. forks, or if the file is removed) still have a working built-in.
- **Path resolution: package-relative, not cwd-relative**. The current `_derive_score_key` uses `os.path.exists(model_spec)` which is cwd-dependent. The launchers `cd` to repo root so this happens to work today, but a proper default needs to resolve relative to the `ifa` package. Helper resolves a default path of `"ifa/models/ifa.onnx"` against the package's parent directory.
- **Verification harness is `scripts/smoke_voice.py record`**, run with `IFA_WAKE_MODEL=ifa/models/ifa.onnx`. This is the same harness we used to validate `hey_mycroft` and to debug `hey_jarvis`/`alexa` failures. It saves a WAV and offline-scores per chunk — perfect for "does this model recognize my voice?" without a full run.
- **Live verification is `scripts/smoke_voice.py transcribe`**, also against the trained model, to confirm the end-to-end voice loop still flows (wake → capture → transcribe → reply, no self-loops, follow-up window working).

## Open Questions

### Resolved During Planning

- **Pronunciation: `EYE-fah`** (IPA /ˈaɪfə/) — like the English letter "I" followed by "fah". Piper TTS positives must use this phonetic reading. Operator should configure Piper voices accordingly in Unit 1's notebook setup (e.g. spelling hints `"eye-fah"` or explicit IPA where supported, NOT plain `"ifa"` which Piper may render as `EE-fah`). If a chosen Piper voice consistently produces a different pronunciation than the user's, swap voices before training — wrong-pronunciation positives are the dominant failure mode for short single-word targets.
- Single-word `"ifa"` is the target (not `"hey ifa"`); `"hey ifa"` becomes Stage 3.5 fallback.
- Synthetic data only for v1, with real-recorded fine-tuning promoted from Deferred to Unit 1's contingency path (per adversarial review).
- Train remotely in Colab, not locally — see Key Technical Decisions.
- Commit `.onnx` to repo (not Releases / LFS / external) — see Key Technical Decisions.
- Make `ifa.onnx` the new default; keep `hey_mycroft` documented as manual fallback.
- Preemptive threshold defaults: 0.8 / consecutive=3. Tuning protocol: threshold first, consecutive second.
- `_DEFAULT_MODEL` resolution: inlined `pathlib.Path(__file__).resolve().parent.parent / "models" / "ifa.onnx"`; missing-file fallback evaluated per-listener inside `_resolve_model_spec`, with a prominent WARNING log when the fallback fires.
- R1 / R2 verification: n=10 sample with ≥9/10 floor (non-regression vs. hey_mycroft); FP-rate gate moved up to Unit 3.

### Deferred to Implementation

- **Exact training hyperparameters** — `auto_train` defaults in the openWakeWord notebook are usually fine. If validation accuracy is < 0.95, the contingency path goes straight to real-recorded fine-tuning (per Scope Boundaries) — do NOT loop on hyperparameter tuning more than once.
- **Number of synthetic positive samples** — notebook default is ~2000. Bump to 3000-5000 if first training round underperforms but only as a side-knob; the primary remediation for poor fit is real recordings.
- **Concrete pinned versions** for openWakeWord, the training notebook, Common Voice, MUSAN, FMA Music, and Piper voices — record in the plan/repo at training time so retraining in 6 months is reproducible.

## Output Structure

```
ifa/
├── models/                       (new — model artifacts; mirrors the ifa/audios/ pattern, no __init__.py)
│   └── ifa.onnx                  (new — committed binary, ~1 MB; score-key "ifa")
└── voice/
    └── wake_word.py              (modified — _DEFAULT_MODEL inlined to bundled path; existence-check moved to per-construction in _resolve_model_spec; missing-file fallback logs WARNING)

docs/
└── plans/
    └── 2026-04-27-001-...-plan.md (this file)

.env.example                      (modified — IFA_WAKE_MODEL default updated, hey_mycroft documented as manual fallback)
README.md                         (modified — voice-mode section: wake word is now "ifa")
```

## Implementation Units

- [ ] **Unit 1: Train `ifa.onnx` in Colab**

**Goal:** Produce a `ifa.onnx` artifact via the openWakeWord training notebook on Colab.

**Requirements:** R1, R2 (preliminary — final verification is in Unit 3)

**Dependencies:** Google account with Colab access. No repo-side dependencies.

**Files:**
- Output: a downloadable `ifa.onnx` (~1 MB) saved locally outside the repo for now (e.g. `~/Downloads/ifa.onnx`).

**Approach:**
- Open https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb in Colab.
- Set the target word to `"ifa"` in the notebook's configuration cell, **pronounced `EYE-fah` (IPA /ˈaɪfə/)** — see Open Questions for rationale. If Piper renders the default spelling as `EE-fah` (the more common English reading), use a phonetic spelling hint (`"eye-fah"`) or an alternate Piper voice that renders the intended pronunciation. Listen to ~5 sample positives BEFORE running training to confirm Piper got it right.
- Use notebook defaults for sample count (~2000), training steps, augmentation, and validation set.
- Run all cells; download the resulting ONNX when done.
- This is operator work, not code work — the notebook does everything.

**Patterns to follow:** None — first time touching the training pipeline. Notebook is self-documenting.

**Test scenarios:** Test expectation: none for this unit — output is a binary artifact verified end-to-end in Unit 3. Notebook reports its own training/validation accuracy and false-positive metrics, which the operator should record (see Verification).

**Verification:**
- Notebook reports validation accuracy ≥ 0.95 and false-positive rate per hour < 1.0 at threshold 0.5.
- ONNX file downloads cleanly (~1 MB, opens in Netron or similar without errors — optional sanity check).
- If validation accuracy < 0.95, retune (more samples, more steps) before proceeding to Unit 2.

---

- [ ] **Unit 2: Commit the model into the repo**

**Goal:** Place `ifa.onnx` at `ifa/models/ifa.onnx`, commit it, and verify the file makes it through git intact.

**Requirements:** R3, R4

**Dependencies:** Unit 1.

**Files:**
- Create: `ifa/models/ifa.onnx` (binary, committed)

**Approach:**
- `mkdir -p ifa/models`, drop the .onnx in it, and commit. Git handles binary files fine; the existing `.gitignore` doesn't exclude it.
- No `__init__.py` is needed: `ifa/audios/sample.wav` is already a committed binary in a sibling directory without one, and we mirror that pattern.
- Sanity check: clone the repo to a fresh tmp dir and verify `ifa/models/ifa.onnx` is present and not corrupted.

**Patterns to follow:**
- The repo doesn't have any precedent for committed binary models, but it does have `ifa/audios/sample.wav` (a committed binary). Same treatment.

**Test scenarios:**
- Edge case: file exists at expected path after commit (`os.path.exists("ifa/models/ifa.onnx")` → True).
- Edge case: file is loadable as ONNX — `onnxruntime.InferenceSession("ifa/models/ifa.onnx")` succeeds without error. Add a test in `ifa/tests/test_wake_word.py` that exercises this once Unit 4 has wired the new default.

**Verification:**
- `git log` shows the commit including the .onnx
- File is byte-identical between repo state and the original Unit 1 download (`shasum` match)

---

- [ ] **Unit 3: Verify `ifa` model fits the user's voice + measure FP rate**

**Goal:** Run `scripts/smoke_voice.py record` against the new model and confirm BOTH (a) the model recognizes the user's voice (R1) and (b) the false-positive rate is acceptable (R2 — moved up from Unit 5 because single-word FP risk is the headline concern with this pivot).

**Requirements:** R1 + R2 (both verified in this unit, not deferred to Unit 5)

**Dependencies:** Unit 2.

**Files:**
- No file changes.
- Activity 1 (hit-rate, R1): from the **repo root** (`cd /path/to/ifa`), run `IFA_WAKE_MODEL=ifa/models/ifa.onnx PYTHONPATH=. python -m scripts.smoke_voice record` on the user's PC. Repeat **10 times** saying "ifa" with normal conversational tone/distance. The relative path requires repo-root cwd; pass an absolute path if running from elsewhere.
- Activity 2 (FP-rate baseline): before measuring `ifa`, measure the **hey_mycroft FP rate** at threshold 0.7 / consecutive=2 over a 30-min window of normal use. This is the comparison number for R2.
- Activity 3 (FP-rate with `ifa`): with `IFA_WAKE_MODEL=ifa/models/ifa.onnx`, threshold 0.8, consecutive=3, listen for 30+ min of normal use (typing, conversation, music). Count unintended wakes.

**Approach:**
- Smoke script's `record` mode saves the capture to `/tmp/ifa_wake_capture.wav` (a single fixed path; on Windows, the equivalent platform temp directory) and prints the offline max score. Open the saved WAV before drawing conclusions about transcription — confirms the audio is actual speech.
- Record offline max scores across the n=10 hit-rate run. Goal: ≥ 0.8 on at least 9 of 10 attempts (matches R1's "≥9/10 non-regression" against hey_mycroft's 10/10).
- **Stop rule for retraining (per R2 contingency):** if max score < 0.6 on >2 attempts in the n=10 run, OR if FP rate > 1/30min at default threshold/consecutive, return to Unit 1 and apply the **real-recorded contingency** (~30-50 user samples) — not "retune notebook hyperparameters and run again." Two retraining cycles maximum before escalating to the Stage 3.5 retreat (`hey ifa`).
- If hit rate passes but FP rate fails, try one round of threshold tuning (raise 0.8 → 0.85, then consecutive 3 → 4) before retraining.

**Patterns to follow:**
- Same harness used to validate `hey_mycroft` — see project memory `project_wake_word.md` for the empirical-verification pattern that prevented shipping `alexa` and `hey_jarvis` blindly.

**Test scenarios:** Test expectation: none — operator-driven empirical verification. The output is two recorded numbers (hit-rate at threshold T / consecutive C, and FP-per-30-min) plus a hey_mycroft FP baseline for the R2 comparison.

**Verification:**
- ≥ 9/10 spoken "ifa" attempts score above the chosen threshold.
- Saved WAVs sound like clear speech (rule out audio capture issues *before* drawing conclusions about training quality).
- FP rate ≤ 1/30min AND ≤ measured hey_mycroft baseline.
- Empirically-found threshold + consecutive values recorded for use in Unit 4's `.env.example` update.

---

- [~] **Unit 4: Make `ifa.onnx` the new default**  *(partial: scaffolding landed in commit X — path-resolution refactor, missing-file fallback with WARNING, fallback_from property, startup-line distinction, .env.example future-state docs. Final flip — pointing `_DEFAULT_MODEL` at the bundled path and the test sweep — gated on Unit 2.)*

**Goal:** Change `_DEFAULT_MODEL` to point at the bundled `.onnx`, with package-relative path resolution.

**Requirements:** R3, R4

**Dependencies:** Unit 2 (file in repo) + Unit 3 (verified working).

**Files:**
- Modify: `ifa/voice/wake_word.py` — `_DEFAULT_MODEL` (inline pathlib resolution) and `_resolve_model_spec` (existence-check fallback).
- Modify: `.env.example` — point default at the bundled path; document `hey_mycroft` as a fallback name.
- Modify: `ifa/tests/test_wake_word.py` — **mechanical sweep**, not a single-line change. Every literal `"hey_mycroft"` reference flips to `"ifa"`: score-key assertions, mock `predict.return_value` keys, mock `predict.side_effect` dict keys, and `download.assert_called_once_with(model_names=["..."])` calls — across init tests, detect tests, mute tests, and the missing-key edge case. The test fixture's score-key seed (`model_inst.predict.return_value = {"hey_mycroft": 0.0}` in `_install_fake_openwakeword`) also flips. Add a separate TestCase that loads `ifa/models/ifa.onnx` directly via `onnxruntime.InferenceSession` (NOT installing the openwakeword fakes) so the bundled binary's integrity is asserted independently of mock state. Note: with the new default being a path (not a built-in name), several existing test scenarios that exercise the "download" path will need a way to assert "no download attempted" instead — adapt them rather than delete.

**Approach:**
- `_DEFAULT_MODEL` is set at module-import time to the absolute path of the bundled model — no new helper function. Inline:
  `_DEFAULT_MODEL = str(pathlib.Path(__file__).resolve().parent.parent / "models" / "ifa.onnx")`. This is package-relative, so it works from any cwd.
- The existing `_resolve_model_spec()` continues to be the single resolution point — no parallel helper. It returns `os.environ.get("IFA_WAKE_MODEL", _DEFAULT_MODEL)` unchanged.
- **Move the missing-file fallback into `_resolve_model_spec` (or its caller)**, NOT module-import time. Per-listener-construction check ensures the fallback re-evaluates if the bundled file is somehow deleted between imports, and lets tests exercise the fallback via `os.path.exists` patching without re-importing the module. Logic: if the resolved spec looks like a path (`os.path.exists`-style probe), and the path doesn't exist, fall back to `"hey_mycroft"` (built-in name) so the listener still boots.
- Existing path-vs-name detection (`os.path.exists`) downstream of `_resolve_model_spec` continues to route paths through openWakeWord's path branch and names through the download branch.

**Patterns to follow:**
- Same `os.environ.get(KEY, default)` pattern used elsewhere in `wake_word.py`.

**Test scenarios:**
- Happy path: with no `IFA_WAKE_MODEL` env var set and the bundled .onnx present, `WakeWordListener` constructs cleanly and `listener.score_key == "ifa"`.
- Edge case: with `IFA_WAKE_MODEL=hey_mycroft` set, the listener uses the built-in name as before — backwards compatibility for users who explicitly opt out.
- Edge case: bundled .onnx file is loadable via `onnxruntime.InferenceSession` (file integrity guard for the committed binary).
- Error path: with the bundled file deleted (simulated via patch on `os.path.exists`) and no env override, `_DEFAULT_MODEL` falls back to `"hey_mycroft"` and the listener constructs cleanly via the built-in name path.

**Verification:**
- All existing wake-word tests pass.
- New "default points at ifa" test asserts `score_key == "ifa"` for default construction.
- Manual check: launch via `run-voice-ps.bat` (or `run-voice.command`) without any `.env` override → startup log shows `wake=ifa`.

---

- [ ] **Unit 5: Live end-to-end verification + docs**

**Goal:** Confirm voice mode works end-to-end with `ifa`, update README, and lock in any tuned threshold.

**Requirements:** R1, R2

**Dependencies:** Unit 4.

**Files:**
- Modify: `README.md` — wake-word section says `"ifa"`.
- Modify: `.env.example` — if Unit 3's tuning landed on a non-default `IFA_WAKE_THRESHOLD` or `IFA_WAKE_CONSECUTIVE`, comment-document it as the recommended starting point (still not setting a non-default value by default in `.env`).
- Update: `~/.claude/projects/-Users-jastagarbrar-Projects-ifa/memory/project_wake_word.md` — flip from "hey_mycroft for v1" to "ifa for v1 (single word, no prefix); hey_mycroft retained as documented manual fallback; hey_ifa documented as Stage 3.5 retreat path if FP rate becomes unacceptable".

**Approach:**
- Operator runs `run-voice-ps.bat` and `run-voice.command`.
- Conduct 5 wake-and-converse turns, mixing single-turn and follow-up patterns. Confirm at least 4/5 detect cleanly.
- Listen for ≥ 30 minutes of normal use (typing, light conversation, music in background) to surface false-positive rate. Target: ≤ 1 unintended wake per 30 minutes. If higher, raise `IFA_WAKE_THRESHOLD` or `IFA_WAKE_CONSECUTIVE` and document the new defaults in `.env.example`.
- Update README's "wake word" sentence from "hey mycroft" to "ifa".
- Update memory file (the closing "How to apply" section) so future sessions see the new state.

**Patterns to follow:**
- Same end-to-end smoke pattern from Stage 2 verification.

**Test scenarios:** Test expectation: none — operator-driven sanity verification. The Unit 4 unit tests already cover code-side correctness.

**Verification:**
- ≥ 4/5 wake attempts succeed
- ≤ 1 false positive in 30 minutes of background use
- README + .env.example + memory all consistent with `ifa` as the default
- 182/182 unit tests still pass

---

## System-Wide Impact

- **Interaction graph:** No runtime interactions change. The same `WakeWordListener` is used; only its default-model spec changes. Voice loop, capture, STT, agent_turn, TTS — all unchanged.
- **Error propagation:** If the bundled .onnx is somehow missing or corrupted, the existing `WakeWordInitError` path catches it and prints an actionable message. The fallback to `"hey_mycroft"` (Unit 4) makes this strictly better than today.
- **State lifecycle risks:** None — the model is loaded once at startup, lives for the process, no rollover or cleanup concerns.
- **API surface parity:** None — env var `IFA_WAKE_MODEL` is the only public surface; the contract (built-in name OR path) is unchanged.
- **Integration coverage:** Unit 4's "bundled file is loadable" test guards against accidental file corruption in commits. Unit 5's manual end-to-end run is the integration coverage.
- **Unchanged invariants:** All Stage 2 voice-mode behavior (mute-during-TTS, follow-up window, capture, transcribe) is preserved exactly. Tool dispatch (Stage 1) is also unchanged.

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **False-positive rate too high** — single-word `"ifa"` has shorter audio context than `"hey jarvis"` etc., so per-frame confidence has more chance to spike on background phonemes | High | High | Preemptive defaults: threshold 0.8 + 3-consecutive frames. **R2 measured in Unit 3, not deferred to Unit 5.** Documented retreat path: `hey ifa` (Stage 3.5) if even tuned single-word can't satisfy R2. |
| **Trained model fits Colab's voice synthesis but not the user's actual voice** (same failure mode that killed `alexa` for this user) | Medium-High | High | Unit 3's offline-WAV verification catches before any code change ships. Contingency path is **Unit 1 retraining with 30-50 real-recorded user samples** — promoted from deferred to in-scope; do not loop on hyperparameter tuning. Cap: 2 retraining cycles before retreating to `hey ifa`. |
| **Pronunciation mismatch between trained-on synthetic samples and the user's actual pronunciation** | Medium | High | Pronunciation must be locked in BEFORE Unit 1 (see Resolve Before Implementation). Piper TTS will produce phonetically distinct positives based on the spelling/IPA the operator feeds it. |
| Committed binary bloats repo or fails to fetch on shallow clones | Low | Low | File is ~1 MB. Repo is currently <10 MB. No LFS, no shallow-clone risk. |
| Training notebook breaks on a future Colab Python or PyTorch update | Low | Medium | Already-shipped `.onnx` keeps working. **Pin specific commit SHAs and dataset versions in the plan/repo at training time** (per adversarial review) so the original training environment is reconstructable. |
| Default-path resolution breaks on Windows due to forward-vs-back slashes | Low | Medium | Unit 4 uses `pathlib.Path` (cross-platform) and `os.path.exists` (cross-platform). String comparison uses resolved absolute paths, not raw env-var strings. |
| **Missing-file fallback to `hey_mycroft` is silent and creates an identity-failure mode** (user gets a working wake word with the wrong name and no obvious indication) | Medium | Medium | Fallback now logs a prominent WARNING and the launcher startup line distinguishes `wake=ifa (bundled)` from `wake=hey_mycroft (FALLBACK — bundled ifa.onnx not found)`. Behavior is observable rather than invisible. |

## Documentation / Operational Notes

- README's voice-mode section (currently: *"The wake word is `hey mycroft` for v1..."*) updates to `ifa` (single word, no prefix).
- `.env.example`'s `IFA_WAKE_MODEL` block updates to show the bundled path as default and `hey_mycroft` (built-in) as a fallback.
- Memory file `project_wake_word.md` updates to reflect the new state.
- No operational rollout, monitoring, or migration concerns — this is a model swap with a hard fallback to a known-good built-in.

## Sources & References

- **Origin plan (Stage 2 Deferred section):** [docs/plans/2026-04-23-003-feat-stage2-voice-input-plan.md](2026-04-23-003-feat-stage2-voice-input-plan.md)
- **Project memory** `project_wake_word.md` — wake-word decision history, identity-fit constraints, false-positive guards
- **openWakeWord training notebook**: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
- **openWakeWord training docs**: https://github.com/dscripka/openWakeWord#training-new-models
- **Piper TTS** (synthetic-data engine used inside the notebook): https://github.com/rhasspy/piper
- **Existing wake-word integration**: `ifa/voice/wake_word.py`, `ifa/tests/test_wake_word.py`
- **Verification harness**: `scripts/smoke_voice.py` (`record` and `transcribe` modes)
