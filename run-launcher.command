#!/usr/bin/env bash
cd "$(dirname "$0")"
if [[ -x venv/bin/python ]]; then PY=venv/bin/python; else PY=python3; fi
exec "$PY" -m launcher "$@"
