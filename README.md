# I F(elt) A(lone)

## What is it:

A bot that you can talk to as ask to do some stuff, there are probably 100s out there and yes this like one of them, but hey, you never know for sure

python 3.12.10

## Running Ifa

### Easy way — double-click

**Text mode** (type to chat):

- **Windows (cmd)**: double-click `run.bat`
- **Windows (PowerShell)**: double-click `run-ps.bat`
- **macOS**: double-click `run.command` (first-time only: right-click → Open to bypass Gatekeeper)

**Voice mode** (say "hey mycroft" to talk):

- **Windows (cmd)**: double-click `run-voice.bat`
- **Windows (PowerShell)**: double-click `run-voice-ps.bat`
- **macOS**: double-click `run-voice.command`

All launchers self-heal: create the Python venv, install dependencies, start Ollama, pull `qwen2.5:7b-instruct` if missing, pre-download the voice models, then launch Ifa. First run takes a few minutes (venv + ~5GB Ollama model + ~500MB voice models); subsequent runs are fast. Voice-mode runtime has `HF_HUB_OFFLINE=1` set, so once the models are cached the app never touches the network. The window always pauses at the end — read any error before closing.

The wake word is `hey mycroft` for v1 (openWakeWord built-in). Custom "hey ifa" training is planned but deferred. Set `IFA_WAKE_MODEL=alexa` / `hey_jarvis` / `hey_rhasspy` to pick a different built-in, or point it at a custom `.onnx` file.

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

- `IFA_WAKE_MODEL` — wake word (default `hey_mycroft`; accepts a path to a custom `.onnx`)
- `IFA_WAKE_THRESHOLD` — wake-word detection threshold 0-1 (default `0.7`)
- `IFA_WHISPER_MODEL` — faster-whisper model name (default `small.en`)
- `IFA_VAD_SILENCE_MS` — trailing silence that ends an utterance (default `1500`)
- `IFA_VAD_MAX_UTTERANCE_MS` — hard cap per turn (default `30000`)

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
