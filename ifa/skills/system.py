from ifa.skills.base import Skill
from ifa.skills.enums import App
from datetime import datetime
import subprocess
import sys
import os


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


class TimeSkill(Skill):
    def can_handle(self, text: str) -> bool:
        return "time" in text.lower()

    def handle(self, text: str) -> str:
        return f"Current time is {datetime.now().strftime('%H:%M')}"


class Application(Skill):

    def __init__(self, tts):
        self.tts = tts

    def open_app(self, app: App, arguments: str) -> str:
        print("Launching Application")

        try:
            if isinstance(app, str):
                app = App(app.lower())

            self.launch_app(app, arguments)
            self.tts.speak(f"Opening {app.value}")
            return f"Opening {app.value}"
        except Exception as e:
            print(e)
            return str(e)

    def launch_app(self, app: App, arguments: str) -> None:
        platform = sys.platform

        print("APP CALLED")
        print(app)
        print(arguments)

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
            if isinstance(cmd, str):
                os.startfile(cmd)
            else:
                subprocess.Popen(cmd)
        else:
            subprocess.Popen(cmd if isinstance(cmd, list) else [cmd])