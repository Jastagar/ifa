# I F(elt) A(lone)

## What is it:

A bot that you can talk to as ask to do some stuff, there are probably 100s out there and yes this like one of them, but hey, you never know for sure

python 3.12.10

## Running Ifa

### Easy way — double-click

- **Windows (cmd)**: double-click `run.bat`
- **Windows (PowerShell)**: double-click `run-ps.bat` (runs `run.ps1` with `-ExecutionPolicy Bypass` so nothing needs configuring)
- **macOS**: double-click `run.command` (first-time only: right-click → Open to bypass Gatekeeper)

All launchers self-heal: create the Python venv, install dependencies, start Ollama, pull `qwen2.5:7b-instruct` if missing, then launch Ifa. First run takes a few minutes (venv + ~5GB model download); subsequent runs are fast. The window always pauses at the end — read any error before closing.

### Prerequisites

- Python 3.12+ on PATH
- [Ollama](https://ollama.com/download) installed

### Manual run (from a terminal)

```bash
PYTHONPATH=. venv/bin/python -m ifa.main       # macOS / Linux
set PYTHONPATH=. && venv\Scripts\python -m ifa.main    # Windows
```

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
