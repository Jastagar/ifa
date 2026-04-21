import subprocess


class TTSService:
    def speak(self, text: str):
        text = text.replace('"', '').replace("'", "")

        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"""
Add-Type -AssemblyName System.Speech;
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$speak.Speak("{text}");
"""
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )