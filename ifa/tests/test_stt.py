from ifa.voice.stt import transcribe

audio_file = "ifa/audios/sample.wav"

text = transcribe(audio_file)
print("Transcribed:", text)