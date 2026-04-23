#!/usr/bin/env bash
# Ifa launcher for macOS. Double-click this file in Finder to start the assistant.
# (macOS opens .command files in Terminal.app directly.)
#
# Self-heals: creates venv if missing, installs requirements, starts Ollama,
# pulls qwen2.5:7b-instruct if not present, then runs python -m ifa.main.

set -u

# Move to the directory this script lives in — double-click from Finder sets
# $PWD to $HOME otherwise.
cd "$(dirname "$0")"

trap 'echo; read -rp "Press Return to close..." _; exit $rc' EXIT

rc=0

die() {
    echo
    echo "*** $* ***"
    rc=1
    exit 1
}

echo
echo "====================================="
echo "  Ifa  -  Personal AI Assistant"
echo "====================================="
echo

# -------- 1. Python venv --------
if [[ ! -x "venv/bin/python" ]]; then
    echo "[setup] venv not found. Creating..."
    if ! command -v python3 >/dev/null 2>&1; then
        die "Python 3 is not installed. Install from https://www.python.org/downloads/ or 'brew install python'."
    fi
    python3 -m venv venv || die "Failed to create venv."
    echo "[setup] Installing dependencies from ifa/requirements.txt ..."
    venv/bin/python -m pip install --upgrade pip >/dev/null
    venv/bin/python -m pip install -r ifa/requirements.txt || die "pip install failed."
fi

# -------- 2. Ollama installed? --------
if ! command -v ollama >/dev/null 2>&1; then
    die "Ollama is not installed. Install with 'brew install ollama' or from https://ollama.com/download, then re-run this script."
fi

# -------- 3. Ollama running? --------
if ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[setup] Ollama is not running. Starting it in the background..."
    # Detach so this script keeps control; logs go to /dev/null.
    nohup ollama serve >/dev/null 2>&1 &
    # Wait up to 15s for the server to come up.
    for _ in $(seq 1 15); do
        sleep 1
        if curl -s -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
            break
        fi
    done
    if ! curl -s -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        die "Ollama did not come up within 15 seconds."
    fi
fi

# -------- 4. qwen2.5:7b-instruct pulled? --------
if ! ollama list 2>/dev/null | grep -qi "qwen2.5:7b-instruct"; then
    echo "[setup] qwen2.5:7b-instruct not found. Pulling now (one-time, ~5GB)..."
    ollama pull qwen2.5:7b-instruct || die "ollama pull failed."
fi

# -------- 5. Launch Ifa --------
echo
echo "[launch] Starting Ifa. Type 'exit' to quit."
echo
PYTHONPATH=. venv/bin/python -m ifa.main
rc=$?

echo
if [[ $rc -ne 0 ]]; then
    echo "Ifa exited with code $rc."
else
    echo "Ifa exited normally."
fi
