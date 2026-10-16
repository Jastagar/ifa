from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class AgentProcess:
    def __init__(self, argv: list[str], cwd: str, max_lines: int = 3000) -> None:
        self.argv = argv
        self.cwd = cwd
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: deque[tuple[int, str]] = deque(maxlen=max_lines)
        self._seq = 0
        self._started_at: float | None = None

    def start(self, clear_env: set[str] | frozenset[str] = frozenset()) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("the agent is already running")
            env = {k: v for k, v in os.environ.items() if k not in clear_env}
            env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONPATH=self.cwd)
            kwargs: dict = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            self._proc = subprocess.Popen(
                self.argv, cwd=self.cwd, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs,
            )
            self._started_at = time.time()
            proc = self._proc
        self._append(f"[launcher] started: {' '.join(self.argv)}")
        threading.Thread(target=self._pump, args=(proc,), daemon=True).start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._append("[launcher] stopping...")
        self._signal(proc, kill=False)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._signal(proc, kill=True)
            proc.wait(timeout=timeout)

    def send(self, line: str) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise RuntimeError("the agent is not running")
        proc.stdin.write(line.rstrip("\r\n") + "\n")
        proc.stdin.flush()

    def status(self) -> dict:
        with self._lock:
            proc = self._proc
            running = proc is not None and proc.poll() is None
            return {
                "state": "running" if running else ("exited" if proc is not None else "stopped"),
                "pid": proc.pid if running else None,
                "exit_code": None if running or proc is None else proc.returncode,
                "started_at": self._started_at if running else None,
            }

    def logs_after(self, seq: int) -> tuple[list[dict], int]:
        with self._lock:
            lines = [{"seq": s, "text": t} for s, t in self._lines if s > seq]
            return lines, self._seq

    def _append(self, text: str) -> None:
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, ANSI.sub("", text)))

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._append(line.rstrip("\r\n"))
        code = proc.wait()
        self._append(f"[launcher] agent exited with code {code}")

    @staticmethod
    def _signal(proc: subprocess.Popen, kill: bool) -> None:
        try:
            if sys.platform == "win32":
                proc.kill() if kill else proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(proc.pid, signal.SIGKILL if kill else signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
