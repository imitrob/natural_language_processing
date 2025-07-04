#!/usr/bin/env python3
import sys, queue, threading, numpy as np, sounddevice as sd, scipy.signal
import time
from faster_whisper import WhisperModel
import soundfile as sf
from playsound import playsound  # pip install playsound==1.3.0

# MODEL        = "distil-large-v3"
# MODEL        = "distil-medium.en"
MODEL        = "distil-small.en"
DEVICE       = "cuda"
DST_SR       = 16_000  # Whisper expects this
WIN_SEC      = 1.0  # analysis window length
OVERLAP_SEC  = 0.4  # context preserved from previous
QUEUE_MAX    = 100

SRC_SR = 16000
CHUNK_IN     = int(SRC_SR * WIN_SEC)  # mic samples per window
TAIL_IN      = int(SRC_SR * OVERLAP_SEC)  # tail to prepend
print(f"Chunk in: {CHUNK_IN} samples, Tail in: {TAIL_IN} samples")
print(SRC_SR, "Hz")

MIC_DEVICE = 2
info   = sd.query_devices()
print(info)
info   = sd.query_devices(MIC_DEVICE, "input")  # Jabra mic input ID
print(info)

class RealtimeSpeechToTextModel():
    def __init__(self):
        super(RealtimeSpeechToTextModel, self).__init__()
        self.audio_q = queue.Queue(maxsize=QUEUE_MAX)
        self.model   = WhisperModel(MODEL, device=DEVICE, compute_type="float32")
        print(f"Loaded model, ready.")

        self.last_received = time.time()
        self.last_loop = time.time()


    def audio_cb(self, indata, frames, t, status):
        if time.time() - self.last_received > 2.0:
            print(f"\n[!] No audio received for {time.time() - self.last_received:.2f} seconds, "
                f"check your microphone connection.", flush=True)
        self.last_received = time.time()
        # if status.input_overflow:
        #     print("Input overflow, dropping data", flush=True)
        #     return
        try:
            self.audio_q.put_nowait(indata.copy())
        except queue.Full:
            pass # drop newest if we’re completely swamped

    def node(self):
        try:
            with sd.InputStream(samplerate=SRC_SR,
                                blocksize=CHUNK_IN,
                                dtype="int16",
                                channels=1,
                                callback=self.audio_cb,
                                device=MIC_DEVICE):
                print("Listening… Ctrl-C to stop")
                worker = threading.Thread(target=self.recogniser, daemon=True)
                worker.start()
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            pass
        finally:
            self.audio_q.put(None)

    def recogniser(self):
        buf = np.empty(dtype=np.int16, shape=[0])
        printed  = "" # running transcript we have shown

        while True:
            data = self.audio_q.get()
            if data is None:  # sentinel → quit
                break
            buf = np.concatenate([buf, data.view(np.int16).squeeze()])

            if time.time() - self.last_loop > 2.0:
                self.last_loop = time.time()
                print(f"\n[+] {len(buf)} samples in buffer, {self.audio_q.qsize()} chunks in queue", flush=True)
            self.last_loop = time.time()
            
            while len(buf) >= CHUNK_IN:
                start_time = time.time()
                window = buf[:CHUNK_IN]
                buf    = buf[CHUNK_IN:]  # drop processed part
                # resample once, scale already done inside to_16k
                # sf.write("output.wav", window, SRC_SR, subtype="PCM_16")
                # playsound("output.wav", block=False)
                # break

                audio16k = to_16k(window)
                segments, _ = self.model.transcribe(
                    audio16k,
                    beam_size=5,
                    language="en",
                    vad_filter=False,
                    temperature=0.0,  # deterministic
                    compression_ratio_threshold=4.0,
                )

                for seg in segments:
                    if seg.text.startswith(printed):
                        new_part = seg.text[len(printed):]
                    else: # dropped context, print full
                        new_part = seg.text
                    if new_part.strip():
                        self.publish_text(new_part, printed)
                        printed += new_part

                print(f"({len(buf)} left)", end="", flush=True)
                buf = np.concatenate([window[-TAIL_IN:], buf])
                print(f"({time.time() - start_time:.2f}s)")

    def publish_text(self, 
                     new_text: str, # newly transcribed text
                     all_text: str, # all text transcribed from the start
                     ):
        print("[["+new_text+"]]", end="", flush=True)

    def test_on_audio_file(self, filename: str = "audio.wav"):
        from scipy.io import wavfile
        from pathlib import Path
        audio_file = Path(filename)
        if not audio_file.exists():
            print(f"Audio file {audio_file} not found.")
            sys.exit(1) 
        # Read the audio file
        SRC_SR, audio_data = wavfile.read(audio_file)
        CHUNK_IN     = int(SRC_SR * WIN_SEC)  # mic samples per window
        TAIL_IN      = int(SRC_SR * OVERLAP_SEC)  # tail to prepend
        print(f"Chunk in: {CHUNK_IN} samples, Tail in: {TAIL_IN} samples")
        print(SRC_SR, "Hz")

        self.audio_q.put_nowait(audio_data.copy()[:len(audio_data)//2])  # start with half a window
        self.audio_q.put_nowait(audio_data.copy()[len(audio_data)//2:])

        self.recogniser()
        
def to_16k(block_i16: np.ndarray) -> np.ndarray:
    f32 = block_i16.astype(np.float32) / 32768.0
    return scipy.signal.resample_poly(f32, DST_SR, SRC_SR)

def main():
    rsst = RealtimeSpeechToTextModel()

    # rsst.test_on_audio_file()
    rsst.node()

if __name__ == "__main__":
    main()