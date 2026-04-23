import base64
import os
import subprocess
import sys
import tempfile
import threading
import time

from rich.console import Console

_console = Console()


class TTSService:
    """Cross-platform text-to-speech via the OS's native speech binaries.

    - **macOS**: `say -o <aiff>` synthesizes to a temp AIFF; `afplay` plays
      it back and blocks on real completion. (`say` alone returns before
      CoreAudio finishes draining, truncating audio on rapid successive
      calls; `afplay` fixes that.)
    - **Windows**: PowerShell's `System.Speech.Synthesis.SpeechSynthesizer`
      via `-EncodedCommand`, with the spoken text passed through an
      environment variable so no metacharacter in `text` can alter
      command structure.
    - **Linux**: `espeak`, with `--` as an argv separator so text starting
      with `-` can't be misread as a flag.

    Thread-safety: each `speak()` call spawns a fresh subprocess, so the
    method is safe to call from daemon threads without additional gating.

    Voice-input cooperation: `is_speaking` is True while a speak() call
    is in flight AND for a short cooldown after it returns. The voice
    wake-word loop reads this on every audio frame and drops audio while
    it's True, so the mic can't self-trigger on Ifa's own output.

    The cooldown is implemented with a monotonic expiry timestamp rather
    than a `threading.Timer`: `Timer.cancel()` does not abort a callback
    that has already begun running, so rapid back-to-back speak() calls
    could expose a brief "mute off" window between the first timer's
    clear-callback firing and the second speak()'s flag-set. The
    timestamp approach has no such window — overlapping calls simply
    extend `_mute_until`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_count = 0  # in-flight speak() calls across all threads
        self._mute_until = 0.0  # monotonic expiry for post-TTS cooldown
        self._cooldown_sec = (
            float(os.environ.get("IFA_TTS_COOLDOWN_MS", "500")) / 1000.0
        )

    @property
    def is_speaking(self) -> bool:
        """True while a speak() call is in flight or within the cooldown window."""
        with self._lock:
            return self._active_count > 0 or time.monotonic() < self._mute_until

    def speak(self, text: str) -> None:
        if not text:
            return

        with self._lock:
            self._active_count += 1
        try:
            if sys.platform == "darwin":
                self._speak_macos(text)
            elif sys.platform == "win32":
                self._speak_windows(text)
            else:
                subprocess.run(["espeak", "--", text], check=False)
        except FileNotFoundError as exc:
            binary = getattr(exc, "filename", None) or "TTS binary"
            _console.print(
                f"[yellow]TTS: `{binary}` not found. Expected "
                "`say`+`afplay` on macOS, `powershell` on Windows, `espeak` on Linux.[/yellow]"
            )
        except Exception as exc:
            _console.print(f"[yellow]TTS failed: {exc}[/yellow]")
        finally:
            with self._lock:
                self._active_count -= 1
                self._mute_until = max(
                    self._mute_until, time.monotonic() + self._cooldown_sec
                )

    def _speak_macos(self, text: str) -> None:
        fd, aiff_path = tempfile.mkstemp(suffix=".aiff", prefix="ifa_tts_")
        os.close(fd)
        try:
            subprocess.run(["say", "-o", aiff_path, "--", text], check=False)
            subprocess.run(["afplay", aiff_path], check=False)
        finally:
            try:
                os.unlink(aiff_path)
            except OSError:
                pass

    def _speak_windows(self, text: str) -> None:
        # String-concat PowerShell commands with user-controlled text are
        # exploitable — a newline in `text` terminates the single-quoted
        # literal in -Command mode and the remainder is parsed as new PS
        # statements. `text` can come from LLM output (prompt injection
        # surface). Instead: pass text out-of-band via an environment
        # variable and use -EncodedCommand so there is no interpolation.
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            '$s.Speak([System.Environment]::GetEnvironmentVariable("IFA_TTS_TEXT"))'
        )
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        env = {**os.environ, "IFA_TTS_TEXT": text}
        subprocess.run(
            ["powershell", "-NoProfile", "-EncodedCommand", encoded],
            env=env,
            check=False,
        )
