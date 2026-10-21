from ifa.skills import browser
from ifa.skills.enums import App
from datetime import datetime
import re
import subprocess
import sys
import os
import tempfile

URI = re.compile(r"^[a-z][a-z0-9+.-]+:\S*$", re.IGNORECASE)


APP_COMMANDS = {
    "win32": {
        App.SPOTIFY: "spotify:",
        App.VS_CODE: "code",
    },
    "darwin": {
        App.SPOTIFY: ["open", "-a", "Spotify"],
        App.VS_CODE: ["open", "-a", "Visual Studio Code"],
    },
    "linux": {
        App.SPOTIFY: ["spotify"],
        App.VS_CODE: ["code"],
    },
}


class TimeSkill:
    def handle(self, text: str) -> str:
        return f"Current time is {datetime.now().strftime('%H:%M')}"


class Application:
    def __init__(self, tts):
        self.tts = tts

    def open_app(self, app: App, arguments: str) -> str:
        print("Launching Application")

        try:
            if isinstance(app, str):
                app = App(app.lower())

            self.launch_app(app, arguments)
            return f"Opening {app.value}"
        except Exception as e:
            print(e)
            return str(e)

    def launch_app(self, app: App, arguments: str) -> None:
        platform = sys.platform

        print("APP CALLED")
        print(f"App: {app}")
        print(f"Arg: {arguments}")

        if platform not in APP_COMMANDS:
            raise ValueError(f"Unsupported platform: {platform}")

        if app in (App.BRAVE, App.PRIVATE_BRAVE):
            url = browser.normalize_url(arguments) if arguments.strip() else ""
            browser.open_url(url, private=app == App.PRIVATE_BRAVE)
            return

        command_map = APP_COMMANDS[platform]

        if app not in command_map:
            if len(arguments) > 0:
                os.startfile(arguments)
                return
            raise ValueError(f"{app} not supported on {platform}")

        cmd = command_map[app]

        if platform == "win32":
            if isinstance(cmd, list):
                subprocess.Popen(cmd + [arguments] if arguments else cmd)
            else:
                os.startfile(cmd)
        else:
            subprocess.Popen(cmd if isinstance(cmd, list) else [cmd])

def launch_custom(command: str, wait_sec: float = 2.0) -> str | None:
    """Start a saved app. Returns None if it started, or what went wrong."""
    command = command.strip()
    target = os.path.expanduser(os.path.expandvars(command.strip('"')))
    if os.path.exists(target) or URI.match(target):
        if sys.platform == "win32":
            os.startfile(target)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", target],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    if sys.platform == "win32":
        # `start` also finds apps that aren't on PATH but are registered with Windows
        # (chrome, discord, excel...), the same way the Run box does.
        if not command.lower().startswith("start "):
            command = f'start "" {command}'
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        kwargs = {"start_new_session": True}
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=errors, **kwargs)
        try:
            code = process.wait(timeout=wait_sec)
        except subprocess.TimeoutExpired:
            return None
        if code == 0:
            return None
        errors.seek(0)
        message = " ".join(errors.read().decode(errors="replace").split())[:300]
    return message or f"the command exited with code {code}"
