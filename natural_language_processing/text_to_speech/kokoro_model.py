
# aplay rather than playsound: playsound plays through GStreamer via python-gi,
# which is not installed in this conda env. Rate conversion is why aplay and not
# sounddevice either -- kokoro generates 24 kHz, the only devices PortAudio sees
# here are raw ALSA hw ones fixed at 48 kHz, and it refuses to resample
# ("Invalid sample rate"). ALSA's plug layer does that for free.
import subprocess
import tempfile

import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000  # what KPipeline generates


def play(audio, samplerate=SAMPLE_RATE):
    """Play one clip, returning when it has finished."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        sf.write(wav.name, audio, samplerate)
        subprocess.run(["aplay", "-q", wav.name], check=True)


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
        for _, _, audio in generator:
            play(audio)


def main():
    cb = Chatterbox()
    cb.speak()

if __name__ == "__main__":
    main()
