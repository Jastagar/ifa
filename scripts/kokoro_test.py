from kokoro import KPipeline
import soundfile as sf
import torch
pipeline = KPipeline(lang_code='a')
text = '''
[Kokoro] I understand the frustration, I just think that it will be a better idea to keep your expectations in check and instead, put your money in mutual funds or something. Do your self a favour
'''
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    sf.write(f'{i}.wav', audio, 24000)