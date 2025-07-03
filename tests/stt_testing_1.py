#!/usr/bin/env python3
"""One-second microphone test

1. Records 1 s from the chosen mic.
2. Saves it as test_chunk.wav (44.1 kHz, 16-bit PCM).
3. Plays the WAV back through the speakers with playsound.
4. Resamples to 16 kHz and feeds it to Whisper.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.signal as sps
from playsound import playsound          # pip install playsound==1.3.0
from faster_whisper import WhisperModel

# ─────────────── CONFIG ──────────────────────────────────────────────
SRC_SR      = 44_100          # your mic’s native rate
DST_SR      = 16_000          # Whisper expects 16 kHz
SECONDS     = 3.0
MIC_DEVICE  = 3               # change to None or another index if needed
OUT_DEVICE  = None            # default speaker
WAV_PATH    = "test_chunk.wav"
MODEL_NAME  = "large-v3"
DEVICE      = "cuda"          # or "cpu"
# ─────────────────────────────────────────────────────────────────────

model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="float16")

def record_one_second() -> np.ndarray:
    print("Recording 1 s …")
    rec = sd.rec(int(SRC_SR * SECONDS),
                 samplerate=SRC_SR,
                 channels=1,
                 dtype='int16',
                 device=MIC_DEVICE)
    sd.wait()
    return rec.squeeze()                # shape (N,) int16

def save_wave(i16: np.ndarray):
    sf.write(WAV_PATH, i16, SRC_SR, subtype="PCM_16")
    print(f"Saved to {WAV_PATH}")

def play_wave():
    print("Playing back …")
    playsound(WAV_PATH, block=True)

def resample_to_16k(i16: np.ndarray) -> np.ndarray:
    f32 = i16.astype(np.float32) / 32768.0
    return sps.resample_poly(f32, DST_SR, SRC_SR)

def transcribe(chunk16k: np.ndarray):
    segs, _ = model.transcribe(chunk16k,
                               language="en",
                               vad_filter=False,
                               temperature=0.0)
    txt = " ".join(s.text for s in segs).strip()
    print("Transcript:", txt or "<empty>")

if __name__ == "__main__":
    chunk_i16 = record_one_second()
    save_wave(chunk_i16)
    play_wave()
    transcribe(resample_to_16k(chunk_i16))
