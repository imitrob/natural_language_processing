#!/usr/bin/env python3
import sys, queue, threading, numpy as np, sounddevice as sd, scipy.signal
from faster_whisper import WhisperModel

MODEL        = "large-v3"
DEVICE       = "cuda"
SRC_SR       = 44_100  # microphone rate
DST_SR       = 16_000  # Whisper expects this
WIN_SEC      = 0.50  # analysis window length
OVERLAP_SEC  = 0.3  # context preserved from previous
CHUNK_IN     = int(SRC_SR * WIN_SEC)  # mic samples per window
TAIL_IN      = int(SRC_SR * OVERLAP_SEC)  # tail to prepend
QUEUE_MAX    = 100

audio_q = queue.Queue(maxsize=QUEUE_MAX)
model   = WhisperModel(MODEL, device=DEVICE, compute_type="float16")
print("Loaded model, ready.")

def audio_cb(indata, frames, t, status):
    if status.input_overflow:
        return
    try:
        audio_q.put_nowait(indata.copy())
    except queue.Full:
        pass # drop newest if we’re completely swamped

def to_16k(block_i16: np.ndarray) -> np.ndarray:
    f32 = block_i16.astype(np.float32) / 32768.0
    return scipy.signal.resample_poly(f32, DST_SR, SRC_SR)

def recogniser():
    buf = np.empty(dtype=np.int16, shape=[0])
    printed  = "" # running transcript we have shown

    while True:
        data = audio_q.get()
        if data is None:  # sentinel → quit
            break
        buf = np.concatenate([buf, data.view(np.int16).squeeze()])

        while len(buf) >= CHUNK_IN:
            window = buf[:CHUNK_IN]
            buf    = buf[CHUNK_IN:]  # drop processed part

            # resample once, scale already done inside to_16k
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
                    print(new_part, end="", flush=True)
                    printed += new_part

            buf = np.concatenate([window[-TAIL_IN:], buf])

try:
    with sd.InputStream(samplerate=SRC_SR,
                        blocksize=CHUNK_IN,
                        dtype="int16",
                        channels=1,
                        callback=audio_cb):
        print("Listening… Ctrl-C to stop")
        worker = threading.Thread(target=recogniser, daemon=True)
        worker.start()
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    pass
finally:
    audio_q.put(None)

