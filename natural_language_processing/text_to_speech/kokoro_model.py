
from playsound import playsound
import soundfile as sf
from kokoro import KPipeline
from IPython.display import display, Audio

class Chatterbox():
    def __init__(self, device="cuda:0"):
        self.pipeline = KPipeline(lang_code='a')
        
    def delete(self):
        del self.pipeline

    def speak(self, text:str = "How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born."):
        generator = self.pipeline(
            text, voice='af_bella',
            speed=1, split_pattern='thiswayitwillneversplit'
        )
        for i, (gs, ps, audio) in enumerate(generator):
            # print(i)  # i => index
            # print(gs) # gs => graphemes/text
            # print(ps) # ps => phonemes
            # display(Audio(data=audio, rate=24000, autoplay=i==0))
            sf.write(f'output.wav', audio, 24000) # save each audio file
            playsound("output.wav", block=True)
            
        
def main():
    cb = Chatterbox()
    cb.speak()

if __name__ == "__main__":
    main()