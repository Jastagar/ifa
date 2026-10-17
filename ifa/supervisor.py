from __future__ import annotations

import os
import subprocess
import sys

RESTART_EXIT_CODE = 75
SUPERVISED_ENV = "IFA_SUPERVISED"


def is_supervised() -> bool:
    return os.environ.get(SUPERVISED_ENV) == "1"


def supervise(argv: list[str] | None = None, run=subprocess.Popen) -> int:
    argv = argv or [sys.executable, "-m", "ifa.main"]
    env = {**os.environ, SUPERVISED_ENV: "1"}
    while True:
        child = run(argv, env=env)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            code = child.wait()
        if code != RESTART_EXIT_CODE:
            return code
        print("\n[ifa] restarting...\n", flush=True)
