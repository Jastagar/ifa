# I F(elt) A(lone)

## What is it:

A bot that you can talk to as ask to do some stuff, there are probably 100s out there and yes this like one of them, but hey, you never know for sure

python 3.12.10

## Launcher (settings + start/stop)

The easiest way to configure and run Ifa:

- **macOS**: double-click `run-launcher.command`
- **Windows**: double-click `run-launcher.bat`
- or: `python -m launcher` (add `--port 8790` / `--no-browser`)

It opens a local page where you can:

- see the detected OS, CUDA GPU, Python venv, and whether Ollama and your model are available
- set up the environment from the **Setup** list: create the venv, install packages (plus the CUDA build of PyTorch when an NVIDIA GPU is found), install and start Ollama, pull the model, download the voice models. Each step shows its exact command (copy it to run it yourself), whether it's already done, and a Run button; **Run missing steps** does them in order and stops at the first failure
- edit every setting (model, GPU/CPU per component, voice, notifications, keys) with validation; saves go to `.env`
- add any other environment variable the agent should get
- add apps Ifa can open by name ("open Discord"): a name plus a program path, a link like `steam://rungameid/570`, or a command. Stored in `apps.json` (not committed)
- launch / stop Ifa and watch its output; in text mode, type to Ifa from the page

Adding a new setting means adding one `Setting(...)` line to `launcher/settings_schema.py`; the page builds itself from that list. The launcher only listens on `127.0.0.1` and never sends secret values back to the browser.

The launcher starts Ifa with `venv`'s Python. On a fresh machine, open the launcher with any Python 3.10+ and press **Run missing steps**. The page only sends step names; the commands themselves are defined in `launcher/setup_steps.py`.

## Running commands

Ifa can run terminal commands (`cmd.exe` on Windows) with the `run_command` tool, for things like "what's my IP" or "how much disk space is left". It never runs one on its own: the first call only proposes the command, Ifa asks, and it runs only if your **very next** message is a yes (yes / okay / go ahead / haan …) within two minutes. That check is in code (`ifa/skills/commands.py`), not just the prompt. Commands that delete recursively, format disks, shut down, change accounts or permissions, or pipe downloaded code into a shell are refused outright. Turn the tool off with `IFA_COMMANDS_ENABLED=0` (launcher: Tools).

## Phone app

The Android app ([ifa.app](https://github.com/jastagar/ifa.app)) connects to Ifa whenever this PC is running Ifa on the same Wi-Fi. It handles the wake word on the phone, a mic button, voice both ways and notifications. On startup Ifa prints:

```
[phone] listening on ws://0.0.0.0:8789/phone (discovery udp/8788). Pairing code: 123456
```

Type that code into the app once. The app finds the PC by broadcasting on UDP 8788 (or you can type the PC's IP). On Windows, allow Python through the firewall on private networks. Phone turns share Ifa's memory with the PC and run one at a time with it. Speech-to-text, the agent and the voice run on the PC, and the reply audio streams back to the phone instead of playing on the PC speakers.

Settings (launcher: Notifications): `IFA_PHONE_ENABLED` (default `1`), `IFA_PHONE_PORT` (`8789`), `IFA_PHONE_DISCOVERY_PORT` (`8788`), `IFA_PHONE_TOKEN` (the pairing code; empty means Ifa generates a 6-digit code and saves it in `.phone_token`). Five wrong codes from one address lock it out for a minute. Code: `ifa/services/phone_server.py`.

## Testing phone notifications

With Ifa running, send fake notifications to its local API (`POST /activate` on port 8787):

- **Windows**: `scripts\notify-test.bat [scenario]`
- **macOS**: `scripts/notify-test.sh [scenario]`

Scenarios: `single` (default: a WhatsApp message from Joe), `burst` (four broken messages from one sender, grouped into one and rated urgent), `urgent` (announced right away), `low` (stored, never announced), `email` (normal priority), `all`. The scripts read `IFA_API_PORT` and `IFA_API_TOKEN` from `.env`.

## Running Ifa

### Easy way — double-click

**Text mode** (type to chat):

- **Windows (cmd)**: double-click `run.bat`
- **Windows (PowerShell)**: double-click `run-ps.bat`
- **macOS**: double-click `run.command` (first-time only: right-click → Open to bypass Gatekeeper)

**Voice mode** (say "ifa" to talk):

- **Windows (cmd)**: double-click `run-voice.bat`
- **Windows (PowerShell)**: double-click `run-voice-ps.bat`
- **macOS**: double-click `run-voice.command`

All launchers self-heal: create the Python venv, install dependencies, start Ollama, pull the model named by `IFA_OLLAMA_MODEL` in `.env` if missing, pre-download the voice models, then launch Ifa. First run takes a few minutes; subsequent runs are fast. Voice-mode runtime has `HF_HUB_OFFLINE=1` set, so once the models are cached the app never touches the network. The window always pauses at the end — read any error before closing.

The wake word is `ifa` for v1 (openWakeWord built-in). Custom "hey ifa" training is planned but deferred. Set `IFA_WAKE_MODEL=alexa` / `hey_jarvis` / `hey_rhasspy` to pick a different built-in, or point it at a custom `.onnx` file.

### Prerequisites

- Python 3.12+ on PATH
- [Ollama](https://ollama.com/download) installed

### Manual run (from a terminal)

```bash
# text mode (default)
PYTHONPATH=. venv/bin/python -m ifa.main                          # macOS / Linux
set PYTHONPATH=. && venv\Scripts\python -m ifa.main               # Windows

# voice mode — wake word + VAD + Whisper
IFA_MODE=voice PYTHONPATH=. venv/bin/python -m ifa.main           # macOS / Linux
set PYTHONPATH=. && set IFA_MODE=voice && venv\Scripts\python -m ifa.main    # Windows
```

Voice mode tunables (env vars):

- `IFA_WAKE_MODEL` — wake word (default `ifa`; accepts a path to a custom `.onnx`)
- `IFA_WAKE_THRESHOLD` — wake-word detection threshold 0-1 (default `0.7`)
- `IFA_WHISPER_MODEL` — faster-whisper model name (default `small.en`)
- `IFA_VAD_SILENCE_MS` — trailing silence that ends an utterance (default `1500`)
- `IFA_VAD_MAX_UTTERANCE_MS` — hard cap per turn (default `30000`)
- `IFA_LISTENING_CUE` — play a brief chime when voice capture begins (default `1`; set `0` to disable)

## Response speed

For the configured Qwen3 model, `IFA_OLLAMA_THINK=0` (the default) skips its
internal reasoning pass for noticeably faster everyday voice replies. Set it
to `1` only when a harder task is worth the extra wait.

`IFA_OLLAMA_KEEP_ALIVE=10m` keeps the local model in GPU memory between turns;
increase it if you use Ifa intermittently and have VRAM to spare.

For quicker voice turns, the local configuration uses `IFA_VAD_SILENCE_MS=900`
and Whisper `base.en`. Switch back to `small.en` or increase the silence value
if a recording is cut short or transcription quality is insufficient.

## n8n integration (optional)

To let Ifa trigger your n8n workflows, copy the example config and edit it:

```bash
cp ifa/config/n8n_workflows.yaml.example ifa/config/n8n_workflows.yaml
# edit URLs + env var names for per-workflow auth
```

The real `ifa/config/n8n_workflows.yaml` is gitignored — webhook URLs contain secret tokens.

## Tests

```bash
PYTHONPATH=. venv/bin/python -m unittest discover ifa/tests
```
