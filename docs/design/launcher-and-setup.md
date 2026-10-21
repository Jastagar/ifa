# Launcher, setup, package versions and offline mode

How Ifa gets installed, configured and started, and why it stays working without internet once it's set up.

**Files:** [launcher/](../../launcher/) (`server.py`, `settings_schema.py`, `setup_steps.py`, `env_file.py`, `agent_process.py`, `system_info.py`, `static/index.html`), [scripts/install_packages.py](../../scripts/install_packages.py), [scripts/setup_voice_models.py](../../scripts/setup_voice_models.py), [ifa/requirements.txt](../../ifa/requirements.txt), [ifa/constraints.txt](../../ifa/constraints.txt), [ifa/utils/offline.py](../../ifa/utils/offline.py), [ifa/supervisor.py](../../ifa/supervisor.py)

---

## Launcher

`python -m launcher` (or `run-launcher.bat` / `.command`) opens a local page:
- an environment check (OS, GPU, the venv and its Python, Ollama, model, phone pairing code)
- Setup steps
- every setting, plus custom variables, apps, prompts and memory
- launch, stop and restart, with the console output and a text box for typing to Ifa

**Decisions and why**
- **One list drives the whole settings page.** Each setting is one `Setting(...)` entry in `settings_schema.py`, and the page, validation and `.env` writes are all generated from it. Adding a setting is one line.
- **`.env` is the only source of settings.** Edits keep comments and unknown keys, and are written atomically. When the launcher starts Ifa, it clears inherited copies of the settings it manages, so `.env` wins over stale shell variables.
- **Security, even though it's localhost:**
  - it binds to 127.0.0.1 only
  - every API call needs a per-session token and a local `Host` header (that blocks DNS-rebinding and cross-site requests)
  - secret values are never sent back to the page, which only shows "Saved"
- **Setup steps are an allowlist.** The page only sends step names; the commands are defined on the server. A page that sends shell strings would be remote code execution with a UI.
- **Each step checks whether it's already done.** Each step checks the actual state of the machine, so **Run missing steps** is safe to press any number of times. Setup and the agent never run at the same time.

## Package versions (2026-10-21)

**The problem:** packages were installed in separate steps (core packages, then Chatterbox, then CUDA torch), each overwriting the last. Re-running "Install core packages" installed `numpy 2.2.6` and `sympy 1.14` over Chatterbox's `numpy<2` and torch's `sympy==1.13.1`, and pip said nothing. torch and transformers broke.

**The fix:**
- **One install command, one lock.** `requirements.txt` lists everything, Chatterbox included, and [ifa/constraints.txt](../../ifa/constraints.txt) pins the exact version of every package (168 on Windows): torch 2.6.0, transformers 5.2.0, numpy 1.26.4, sympy 1.13.1, huggingface-hub 1.11.0…
- **Python 3.10, 3.11 and 3.12.** The first lock came from a 3.12 venv and pinned packages that need 3.11+ (scikit-learn 1.8.0, networkx 3.7), so pip on an older venv failed. The lock is now resolved for 3.10 (with `uv pip compile --python-platform windows --python-version 3.10`) while keeping the core versions. It's verified to resolve on Windows 3.10, 3.11 and 3.12 and macOS 3.12. It was also installed into the Mac's venv: `pip check` is clean, tests pass, and Whisper loads.
- **`scripts/install_packages.py`** (launcher **Install packages**, and the `run.*` scripts):
  1. prints the venv's Python
  2. upgrades pip (old pips give up on a dependency set this size)
  3. installs `-r requirements.txt -c constraints.txt`
  4. runs `pip check`
- **The launcher compares installed versions with the lock.** If something drifts, the step shows "Wrong versions: numpy 2.2.6 (needs 1.26.4)", and running it repairs the venv. A CUDA torch (`2.6.0+cu124`) counts as 2.6.0.
- **Rule:** change versions only in `constraints.txt`, never with one-off `pip install`s. Regenerate the lock with the `uv pip compile` command in its header.
- **CUDA:** **Switch PyTorch to the CUDA build** reinstalls the same torch version from the cu124 index, without dependencies.

## Offline mode (2026-10-21)

**Goal:** once packages and models are downloaded, Ifa runs with no internet.

**What was wrong**, found by blocking every outside connection and logging who tried:
- Whisper and Chatterbox checked the Hugging Face Hub on every start. `run.bat` even forced `HF_HUB_OFFLINE=0`. With no internet, Whisper loading **crashed**.
- Chatterbox was never pre-downloaded, so the first launch fetched it.
- Importing `pywhatkit` pinged google.com and raised an error offline.

**The fix:**
- **[ifa/utils/offline.py](../../ifa/utils/offline.py)** runs at startup, before any Hugging Face library loads. If the Whisper and Chatterbox models are in the cache, it sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; if not, it stays online and names what's missing. `IFA_OFFLINE=auto|on|off` (launcher: Advanced) overrides this.
- **Download voice models** now fetches the wake word, Whisper, Chatterbox and the speaker model.
- **`pywhatkit` is imported only when sending a WhatsApp message.**
- **Verified with the network guard:** startup, Whisper, all 17 tools, the wake word and the voice check make **0 network attempts**. Chatterbox's loading call returned the cached model with 0 attempts.
- **Still needs the internet** (and fails with a clear error offline): Spotify, YouTube and web search, sending WhatsApp messages, n8n.

## Running and restarting

- `python -m ifa.main` runs under a small supervisor that restarts Ifa when it exits with code 75 (`restart_ifa`, other session).
- The launcher's stop button kills the whole process tree.
- `run.bat` / `run.ps1` / `run.command` still work for a terminal-only start.

## Next

- Make Ifa runnable on Apple Silicon: replace `blingfire` (Intel-only on macOS) and add an `mps` device option.
- A one-click "rebuild the venv" for when its Python is wrong. On Windows the launcher can't delete the venv it's running from, so it would have to run from the system Python.
