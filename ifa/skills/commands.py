from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CONFIRM_WINDOW_SEC = 120
MAX_OUTPUT_CHARS = 1500

YES = re.compile(
    r"^(yes|yeah|yep|yup|ya|sure|ok|okay|confirm(ed)?|go ahead|do it|run it|please do|haan?|ha+|han ji|haan ji|"
    r"ji|theek hai|thik hai|kar do|chala do)\b"
)
NOT = re.compile(r"\b(no|not|don'?t|dont|wait|stop|cancel|mat|nahi|nahin)\b")

DANGEROUS = [
    (r"\bformat(\.com)?\s+[a-z]:", "formats a drive"),
    (r"\bdiskpart\b", "edits disk partitions"),
    (r"\b(del|erase)\b.*\s/s\b", "deletes files recursively"),
    (r"\b(rd|rmdir)\b.*\s/s\b", "deletes folders recursively"),
    (r"\brm\s+(-\S*\s+)*-\S*[rR]", "deletes files recursively"),
    (r"\bremove-item\b.*-recurse", "deletes files recursively"),
    (r"\b(shutdown|restart-computer|stop-computer|reboot|halt|poweroff)\b", "shuts down or restarts the computer"),
    (r"\breg(\.exe)?\s+delete\b", "deletes registry keys"),
    (r"\b(bcdedit|mkfs\S*|fdisk|vssadmin|cipher\s+/w)\b", "changes disks or boot settings"),
    (r"\bdd\s+if=", "writes raw disk data"),
    (r":\(\)\s*\{", "is a fork bomb"),
    (r"\b(takeown|icacls|chmod\s+-r|chown\s+-r)\b", "changes file ownership or permissions"),
    (r"\bnet\s+(user|localgroup)\b", "changes user accounts"),
    (r"\bset-executionpolicy\b", "changes the script security policy"),
    (r"\b(invoke-expression|iex)\b|\|\s*(sh|bash|zsh|cmd|powershell|pwsh)\b", "runs downloaded or generated code"),
    (r"\bsudo\b|\brunas\b", "runs as administrator"),
]


def is_yes(text: str) -> bool:
    words = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    return bool(YES.match(words)) and not NOT.search(words)


def dangerous(command: str) -> str | None:
    lowered = command.lower()
    for pattern, reason in DANGEROUS:
        if re.search(pattern, lowered):
            return reason
    return None


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()).lower() == " ".join(b.split()).lower()


@dataclass
class Pending:
    command: str
    turn: int
    at: float


class CommandRunner:
    def __init__(self, timeout: float = 30, clock=time.monotonic, popen=subprocess.Popen) -> None:
        self.timeout = timeout
        self._clock = clock
        self._popen = popen
        self.pending: Pending | None = None

    def handle(self, command: str, turn: int, user_text: str) -> str:
        command = command.strip()
        if not command or "\n" in command or "\r" in command:
            return "ERROR: give exactly one single-line command."
        reason = dangerous(command)
        if reason:
            self.pending = None
            return (f"REFUSED: `{command}` {reason}, so Ifa will not run it. "
                    "Tell Sir he can run it himself if he really wants to.")
        p = self.pending
        if (p and _same(p.command, command) and turn == p.turn + 1
                and self._clock() - p.at <= CONFIRM_WINDOW_SEC and is_yes(user_text)):
            self.pending = None
            return self._execute(command)
        self.pending = Pending(command, turn, self._clock())
        return (f"NOT RUN YET. In one short sentence, tell Sir what `{command}` will do and ask whether "
                "to run it. Do not call run_command again in this turn. If his next message is a yes, "
                "call run_command again with exactly the same command.")

    def _execute(self, command: str) -> str:
        print(f"[command] running: {command}")
        kwargs: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" \
            else {"start_new_session": True}
        try:
            proc = self._popen(command, shell=True, cwd=str(Path.home()), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                               **kwargs)
        except OSError as exc:
            return f"ERROR: could not start the command: {exc}"
        try:
            output, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            output, _ = proc.communicate()
            return (f"Ran `{command}` but it was still going after {self.timeout:g} seconds, so it was stopped.\n"
                    f"Output so far:\n{_tail(output)}")
        print(f"[command] exit code {proc.returncode}")
        return f"Ran `{command}`. Exit code {proc.returncode}.\nOutput:\n{_tail(output)}"


def _tail(output: str | None) -> str:
    output = (output or "").strip()
    if not output:
        return "(no output)"
    if len(output) > MAX_OUTPUT_CHARS:
        return "(earlier output cut)\n" + output[-MAX_OUTPUT_CHARS:]
    return output


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()
