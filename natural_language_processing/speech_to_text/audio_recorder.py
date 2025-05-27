import time
import wave
from playsound import playsound
import subprocess
import audioop       # std-lib on Linux; part of CPython
from pathlib import Path

popelkapc_command = ["pasuspender", "--", "arecord", 
    f"-D", f"plughw:1,0",
    "-f", "cd",
    "-t", "wav",
]
homepc_command = [
    'arecord', 
    # '-D', 'plughw:3,0', 
    '-f', 'S16_LE', '-r', '24000', '-c', '1', '-t', 'wav',
]
thinkpad_command = [
    # "pasuspender", "--",
    "arecord",
    "-f", "cd",
    "-t", "wav",
    f"-D", f"plughw:3,0"
]

class AudioRecorder():
    def __init__(self):
        self.is_recording = False
        self.process = None

    def start_recording(self, 
                        output_file=f"recording_{int(time.time()*100)}.wav", 
                        duration: int = 5, # maximum duration 
                        ):
        self.is_recording = True
        self.duration = duration
        self.output_file = output_file


        self.start_time = time.time()
        self.process = subprocess.Popen(
            thinkpad_command + [
                "-d", str(duration), # Maximum record duration
                output_file,
            ])
        
    def stop_recording(self):
        while (time.time() - self.start_time) < 1.0: # It should record at least for a second 
            time.sleep(0.1)
        self.process.terminate()  # Send SIGTERM to arecord
        self.process.wait()  # Wait for process to exit

        self.is_recording = False
        if not self.check_sound(self.output_file, SILENCE_RMS_THRESHOLD=100):
            print("WARNING YOUR MIC MIGHT BE OFF!", flush=True)
        return self.output_file, self.start_time

    @classmethod
    def check_sound(cls, soundfile: str | Path, SILENCE_RMS_THRESHOLD: int) -> bool:
        """Return overall RMS of a WAV file.
        """
        with wave.open(str(soundfile), "rb") as wf:
            frames = wf.readframes(wf.getnframes())

            RMS = audioop.rms(frames, wf.getsampwidth())
            return bool(RMS > SILENCE_RMS_THRESHOLD)


if __name__ == "__main__":
    rec = AudioRecorder()
    rec.start_recording()
    time.sleep(5)
    file,stamp = rec.stop_recording()
    print("Saved as: ", file, stamp)
    try:
        print("Playing the sound")
        playsound(file)
    except:
        print("File not recorded!")
