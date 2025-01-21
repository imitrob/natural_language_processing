# import sounddevice as sd
import numpy as np
# import scipy.io.wavfile as wav
import threading
import time
# import subprocess


# class SoundDeviceRecorder():
#     def __init__(self,
#                  fs=44100, # Sample rate
#                 ):
#         # Global variables
#         self.is_recording = False
#         self.recording_thread = None
#         self.fs = fs
#         self.recorded_data = []

#     def record_audio(self):
#         self.recorded_data = []  # Reset recorded data before starting

#         def callback(indata, frames, time, status):
#             if status:
#                 print(status)
#             self.recorded_data.append(indata.copy())

#         with sd.InputStream(samplerate=self.fs, channels=2, callback=callback):
#             while self.is_recording:
#                 sd.sleep(100)
    
#     def start_recording(self, duration=5):
#         if not self.is_recording:
#             self.is_recording = True
#             self.recording_thread = threading.Thread(target=self.record_audio)
#             self.recording_thread.start()
#             print("Recording started...")
    
#     def stop_recording(self):
#         if self.is_recording:
#             self.is_recording = False
#             self.recording_thread.join()  # Wait for recording thread to finish
#             print("Recording stopped...")
#             file_name = self.save_recording()
#             return file_name
#         return None
    
#     def save_recording(self):
#         audio_data = np.concatenate(self.recorded_data, axis=0)  # Combine recorded data
#         file_name = f"recording_{int(time.time())}.wav"
#         wav.write(file_name, self.fs, audio_data)
#         print(f"Recording saved to {file_name}")
#         return file_name

# class PopelkaRecorder(): # Working on Popelka-PC
#     def start_recording(output_file="recording.wav", duration=5, sound_card=1):
#         command = [
#             "pasuspender", "--", "arecord", 
#             f"-D", f"plughw:{sound_card},0",
#             "-f", "cd",
#             "-t", "wav",
#             "-d", str(duration),
#             output_file
#         ]
        
#         subprocess.run(command, check=True)

import pyaudio
import wave

class PyAudioRecorder():
    FORMAT = pyaudio.paInt32  # 16-bit audio format
    CHANNELS = 1             # Mono
    CHUNK = 1024             # Buffer size
    OUTPUT_FILE = "output.wav"

    def __init__(self,
                 fs=44100, # Sample rate
                ):
        # Global variables
        self.is_recording = False
        self.recording_thread = None
        self.fs = fs
        self.recorded_data = []

        self.stream = None
        self.frames = []

    def record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=self.CHANNELS,
                        rate=self.fs, input=True,
                        frames_per_buffer=self.CHUNK)
        
        try:
            while self.is_recording:
                data = stream.read(self.CHUNK)
                self.frames.append(data)
        except KeyboardInterrupt:
            print("Recording stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save audio to a file
            with wave.open(self.OUTPUT_FILE, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(p.get_sample_size(self.FORMAT))
                wf.setframerate(self.fs)
                wf.writeframes(b''.join(self.frames))

    def start_recording(self):
        print(".")
        if not self.is_recording:
            print("start recordgin")

            self.is_recording = True
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            print("Recording started...")
    
    def stop_recording(self):
        print("...")
        

        if self.is_recording:
            print("stop recordgin")

            self.is_recording = False
            self.recording_thread.join()  # Wait for recording thread to finish
            print("Recording stopped...")
            # file_name = self.save_recording()
            return self.OUTPUT_FILE
        return None

AudioRecorder = PyAudioRecorder

if __name__ == "__main__":
    rec = AudioRecorder()
    rec.start_recording()
    time.sleep(2)
    file = rec.stop_recording()
    print("Saved as: ", file)
