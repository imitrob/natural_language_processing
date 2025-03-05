import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import time
# import subprocess


class SoundDeviceRecorder():
    def __init__(self,
                 fs=44100, # Sample rate
                ):
        # Global variables
        self.is_recording = False
        self.recording_thread = None
        self.fs = fs
        self.recorded_data = []

    def record_audio(self):
        self.recorded_data = []  # Reset recorded data before starting

        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.recorded_data.append(indata.copy())

        with sd.InputStream(samplerate=self.fs, channels=1, callback=callback):
            while self.is_recording:
                sd.sleep(100)
    
    def start_recording(self, duration=5):
        if not self.is_recording:
            self.is_recording = True
            self.start_time = time.time()
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            print("Recording started...")
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.recording_thread.join()  # Wait for recording thread to finish
            print("Recording stopped...")
            file_name = self.save_recording()
            return file_name, self.start_time
        return None
    
    def save_recording(self):
        audio_data = np.concatenate(self.recorded_data, axis=0)  # Combine recorded data
        file_name = f"recording_{int(time.time()*100)}.wav"
        wav.write(file_name, self.fs, audio_data)
        print(f"Recording saved to {file_name}")
        return file_name

# class PopelkaRecorder(): # Working on Popelka-PC
#     def start_recording(output_file=f"recording_{int(time.time()*100)}.wav", duration=5, sound_card=1):
#         command = [
#             "pasuspender", "--", "arecord", 
#             f"-D", f"plughw:{sound_card},0",
#             "-f", "cd",
#             "-t", "wav",
#             "-d", str(duration),
#             output_file
#         ]
        
#         subprocess.run(command, check=True)

import subprocess, time

class HomeRecorder():
    def __init__(self):
        self.duration = 5
        self.is_recording = False
    def start_recording(self, output_file=f"recording_{int(time.time()*100)}.wav", duration=5, sound_card=3):
        self.is_recording = True
        self.duration = duration
        self.output_file = output_file
        command = [
            'arecord', '-D', 'plughw:3,0', '-f', 'S16_LE', '-r', '24000', '-c', '1', '-t', 'wav', '-d', '5',# 'recording.wav'
            # 'arecord', '-D', 'plughw:3,0', '-f', 'cd', '-t', 'wav', '-d', '5', 
            output_file
        ]
        self.start_time = time.time()
        
        subprocess.run(command, check=True)
        self.is_recording = False
    
    def stop_recording(self):
        while self.is_recording:
            time.sleep(0.1)
        return self.output_file, self.start_time

class VisionRecorder():
    def __init__(self):
        self.duration = 5
        self.is_recording = False
    def start_recording(self, output_file=f"recording_{int(time.time()*100)}.wav", duration=5, sound_card=3):
        self.is_recording = True
        self.duration = duration
        self.output_file = output_file
        command = [
            "pasuspender", "--", "arecord", 
            f"-D", f"plughw:{sound_card},0",
            "-f", "cd",
            "-t", "wav",
            "-d", str(duration),
            output_file
        ]
        self.start_time = time.time()
        
        subprocess.run(command, check=True)
        self.is_recording = False
    
    def stop_recording(self):
        while self.is_recording:
            time.sleep(0.1)
        return self.output_file, self.start_time

import pyaudio
import wave

class PyAudioRecorder():
    FORMAT = pyaudio.paInt32  # 16-bit audio format
    CHANNELS = 2             # Mono
    CHUNK = 1024             # Buffer size
    OUTPUT_FILE = f"recording_{int(time.time()*100)}.wav"

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
            print("Recording stopped.", flush=True)
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
        if not self.is_recording:
            self.is_recording = True
            self.start_time = time.time()
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            print("Recording started...", flush=True)
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.recording_thread.join()  # Wait for recording thread to finish
            print(f"Recording stopped... with file: {self.OUTPUT_FILE}", flush=True)
            # file_name = self.save_recording()
            return self.OUTPUT_FILE, self.start_time
        return None

# Choose the recorder that works for you
AudioRecorder = VisionRecorder

if __name__ == "__main__":
    rec = AudioRecorder()
    rec.start_recording()
    time.sleep(5)
    file = rec.stop_recording()
    print("Saved as: ", file)


