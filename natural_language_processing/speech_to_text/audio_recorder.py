import os
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
    f"-D", f"plughw:2,0"
]

# ALSA card indices are assignment-order, so plughw:2,0 points at a different
# card after a reboot or a replug. The card *id* is stable: `arecord -l` prints
# it in brackets (Jabra Speak 710 -> J710). Override with AUDIO_DEVICE if the
# machine has other hardware.
jabra_command = [
    "arecord",
    "-D", os.environ.get("AUDIO_DEVICE", "plughw:CARD=J710,DEV=0"),
    "-f", "cd",
    "-t", "wav",
]

class AudioRecorder():
    def __init__(self, cmd=jabra_command):
        self.is_recording = False
        self.process = None
        self.cmd = cmd

    def start_recording(self,
                        output_file=None,
                        duration: int = 5, # maximum duration
                        ):
        # The name is built per call, not as a default argument: a default is
        # evaluated once at import, so every recording of a session reused one
        # filename. Absolute, because the path travels to the speech-to-text
        # server over ROS and that process has its own working directory.
        if output_file is None:
            output_file = f"recording_{time.time_ns()}.wav"
        self.is_recording = True
        self.duration = duration
        self.output_file = str(Path(output_file).resolve())


        self.start_time = time.time()
        self.process = subprocess.Popen(
            self.cmd + [
                "-d", str(duration), # Maximum record duration
                self.output_file,
            ])
        
    def stop_recording(self):
        while (time.time() - self.start_time) < 1.0: # It should record at least for a second 
            time.sleep(0.1)
        self.process.terminate()  # Send SIGTERM to arecord
        self.process.wait()  # Wait for process to exit

        self.is_recording = False
        # if not self.check_sound(self.output_file, SILENCE_RMS_THRESHOLD=100):
        #     print("WARNING YOUR MIC MIGHT BE OFF!", flush=True)
        if not Path(self.output_file).is_file():
            # arecord wrote nothing: no microphone, or the wrong -D card. Callers
            # treat None as "no recording"; announcing a file that does not exist
            # only moves the failure into whoever opens it.
            print(f"No recording made, `{' '.join(self.cmd)}` wrote no file. "
                  f"Is a microphone connected? Check the `-D plughw:` card in "
                  f"audio_recorder.py (`arecord -l` lists them).", flush=True)
            return None, self.start_time
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
    # No-microphone path: arecord fails, writes no file, stop_recording reports None.
    rec = AudioRecorder(cmd=["arecord", "-D", "plughw:99,0", "-f", "cd", "-t", "wav"])
    rec.start_recording(output_file="/tmp/_audio_recorder_selfcheck.wav")
    assert rec.stop_recording()[0] is None
    print("no-microphone check ok")

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
