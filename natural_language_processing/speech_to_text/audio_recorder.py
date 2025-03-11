import time
from playsound import playsound
import subprocess

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
    f"-D", f"plughw:1,0"
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
        self.process.terminate()  # Send SIGTERM to arecord
        self.process.wait()  # Wait for process to exit

        self.is_recording = False
        return self.output_file, self.start_time

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
