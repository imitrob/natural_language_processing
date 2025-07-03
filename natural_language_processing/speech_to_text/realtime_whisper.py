#!/usr/bin/env python3
import sys, queue, threading, numpy as np, sounddevice as sd, scipy.signal
import time
from faster_whisper import WhisperModel
import soundfile as sf
from playsound import playsound  # pip install playsound==1.3.0

MODEL        = "distil-large-v3"
MODEL        = "distil-medium.en"
MODEL        = "distil-small.en"
DEVICE       = "cuda"
DST_SR       = 16_000  # Whisper expects this
WIN_SEC      = 1.0  # analysis window length
OVERLAP_SEC  = 0.4  # context preserved from previous
QUEUE_MAX    = 100

audio_q = queue.Queue(maxsize=QUEUE_MAX)
model   = WhisperModel(MODEL, device=DEVICE, compute_type="float32")
print(f"Loaded model, ready.")

last_received = time.time()
last_loop = time.time()

def audio_cb(indata, frames, t, status):
    global last_received
    if time.time() - last_received > 2.0:
        print(f"\n[!] No audio received for {time.time() - last_received:.2f} seconds, "
              f"check your microphone connection.", flush=True)
    last_received = time.time()
    # if status.input_overflow:
    #     print("Input overflow, dropping data", flush=True)
    #     return
    try:
        audio_q.put_nowait(indata.copy())
    except queue.Full:
        pass # drop newest if we’re completely swamped

def to_16k(block_i16: np.ndarray) -> np.ndarray:
    f32 = block_i16.astype(np.float32) / 32768.0
    return scipy.signal.resample_poly(f32, DST_SR, SRC_SR)

def recogniser():
    global last_loop
    buf = np.empty(dtype=np.int16, shape=[0])
    printed  = "" # running transcript we have shown

    while True:
        data = audio_q.get()
        if data is None:  # sentinel → quit
            break
        buf = np.concatenate([buf, data.view(np.int16).squeeze()])

        if time.time() - last_loop > 2.0:
            last_loop = time.time()
            print(f"\n[+] {len(buf)} samples in buffer, {audio_q.qsize()} chunks in queue", flush=True)
        last_loop = time.time()
        
        while len(buf) >= CHUNK_IN:
            start_time = time.time()
            window = buf[:CHUNK_IN]
            buf    = buf[CHUNK_IN:]  # drop processed part
            # resample once, scale already done inside to_16k
            # sf.write("output.wav", window, SRC_SR, subtype="PCM_16")
            # playsound("output.wav", block=False)
            # break

            audio16k = to_16k(window)
            segments, _ = model.transcribe(
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
                    print("[["+new_part+"]]", end="", flush=True)
                    printed += new_part

            print(f"({len(buf)} left)", end="", flush=True)
            buf = np.concatenate([window[-TAIL_IN:], buf])
            print(f"({time.time() - start_time:.2f}s)")





TEST_FILE = False  # set to True to test with a file instead of mic input
if TEST_FILE:
    from scipy.io import wavfile
    from pathlib import Path
    audio_file = Path("audio.wav")
    if not audio_file.exists():
        print(f"Audio file {audio_file} not found.")
        sys.exit(1) 
    # Read the audio file
    SRC_SR, audio_data = wavfile.read(audio_file)
    CHUNK_IN     = int(SRC_SR * WIN_SEC)  # mic samples per window
    TAIL_IN      = int(SRC_SR * OVERLAP_SEC)  # tail to prepend
    print(f"Chunk in: {CHUNK_IN} samples, Tail in: {TAIL_IN} samples")
    print(SRC_SR, "Hz")

    audio_q.put_nowait(audio_data.copy()[:len(audio_data)//2])  # start with half a window
    audio_q.put_nowait(audio_data.copy()[len(audio_data)//2:])

    recogniser()
    exit()

try:
    SRC_SR = 16000
    CHUNK_IN     = int(SRC_SR * WIN_SEC)  # mic samples per window
    TAIL_IN      = int(SRC_SR * OVERLAP_SEC)  # tail to prepend
    print(f"Chunk in: {CHUNK_IN} samples, Tail in: {TAIL_IN} samples")
    print(SRC_SR, "Hz")


    info   = sd.query_devices()
    print(info)
    info   = sd.query_devices(7, "input")  # Jabra mic input ID
    print(info)
    

    with sd.InputStream(samplerate=SRC_SR,
                        blocksize=CHUNK_IN,
                        dtype="int16",
                        channels=1,
                        callback=audio_cb,
                        device=7):
        print("Listening… Ctrl-C to stop")
        worker = threading.Thread(target=recogniser, daemon=True)
        worker.start()
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    pass
finally:
    audio_q.put(None)

