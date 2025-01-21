
from natural_language_processing.text_to_speech.Kokoro.models import build_model
from natural_language_processing.text_to_speech.Kokoro.kokoro import generate
import natural_language_processing
import torch
import sounddevice as sd
from IPython.display import display, Audio

class Chatterbox():
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = build_model(f'{natural_language_processing.tts_path}/kokoro-v0_19.pth', device)
        self.voice_name = [
            'af', # Default voice is a 50-50 mix of Bella & Sarah
            'af_bella', 'af_sarah', 'am_adam', 'am_michael',
            'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
            'af_nicole', 'af_sky',
        ][0]
        self.voicepack = torch.load(f'{natural_language_processing.tts_path}/voices/{self.voice_name}.pt', weights_only=True).to(device)
        print(f'Loaded voice: {self.voice_name}')
        
    def speak(self, text:str = "How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born."):
        audio, out_ps = generate(self.model, text, self.voicepack, lang=self.voice_name[0])

        display(Audio(data=audio, rate=24000, autoplay=True))
        print(out_ps)
        
        sd.play(audio, samplerate=24000)
        sd.wait()  # Wait until the audio finishes playing


