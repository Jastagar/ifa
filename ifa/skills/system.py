from ifa.skills.enums import App
from datetime import datetime
import re
import subprocess
import sys
import os

URI = re.compile(r"^[a-z][a-z0-9+.-]+:\S*$", re.IGNORECASE)


APP_COMMANDS = {
    "win32": {
        App.SPOTIFY: "spotify:",
        App.VS_CODE: "code",
        App.BRAVE: "C:/Progra~1/BraveSoftware/Brave-Browser/Application/brave.exe",
        App.PRIVATE_BRAVE: ["C:/Progra~1/BraveSoftware/Brave-Browser/Application/brave.exe","--incognito"],
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

def launch_custom(command: str) -> None:
    target = os.path.expanduser(os.path.expandvars(command.strip().strip('"')))
    if os.path.exists(target) or URI.match(target):
        if sys.platform == "win32":
            os.startfile(target)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", target],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        subprocess.Popen(command, shell=True, creationflags=flags,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(command, shell=True, start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
