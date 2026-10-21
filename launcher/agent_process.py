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


class LogBuffer:
    def __init__(self, max_lines: int = 3000, log_dir=None) -> None:
        self._lock = threading.Lock()
        self._lines: deque[tuple[int, str]] = deque(maxlen=max_lines)
        self._seq = 0
        self.log_dir = log_dir

    def append(self, text: str) -> None:
        clean = ANSI.sub("", text)
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, clean))
            if self.log_dir is not None:
                try:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%H:%M:%S")
                    with open(self.log_dir / f"launcher-{time.strftime('%Y-%m-%d')}.log", "a", encoding="utf-8") as f:
                        f.write(f"{stamp} {clean}\n")
                except OSError:
                    pass

    def text(self) -> str:
        with self._lock:
            return "\n".join(t for _, t in self._lines)

    def after(self, seq: int) -> tuple[list[dict], int]:
        with self._lock:
            return [{"seq": s, "text": t} for s, t in self._lines if s > seq], self._seq


class AgentProcess:
    def __init__(self, argv: list[str], cwd: str, log: LogBuffer | None = None, name: str = "agent") -> None:
        self.argv = argv
        self.cwd = cwd
        self.name = name
        self.log = log or LogBuffer()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._started_at: float | None = None

    def start(
        self,
        clear_env: set[str] | frozenset[str] = frozenset(),
        argv: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError(f"the {self.name} is already running")
            if argv is not None:
                self.argv = argv
            env = {k: v for k, v in os.environ.items() if k not in clear_env}
            env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONPATH=self.cwd)
            env.update(extra_env or {})
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
            raise RuntimeError(f"the {self.name} is not running")
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

    def wait(self) -> int | None:
        with self._lock:
            proc = self._proc
        return None if proc is None else proc.wait()

    def logs_after(self, seq: int) -> tuple[list[dict], int]:
        return self.log.after(seq)

    def _append(self, text: str) -> None:
        self.log.append(text)

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._append(line.rstrip("\r\n"))
        code = proc.wait()
        for pipe in (proc.stdin, proc.stdout):
            try:
                if pipe:
                    pipe.close()
            except OSError:
                pass
        self._append(f"[launcher] {self.name} exited with code {code}")

    @staticmethod
    def _signal(proc: subprocess.Popen, kill: bool) -> None:
        try:
            if sys.platform == "win32":
                if kill:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    proc.kill()
                else:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(proc.pid, signal.SIGKILL if kill else signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
