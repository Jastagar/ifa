import pyttsx3

class PyttsxEngine:
    def speak(self, text: str):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()