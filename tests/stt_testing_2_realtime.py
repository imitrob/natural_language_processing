#!/usr/bin/env python3
import sys, numpy as np
from faster_whisper import WhisperModel
import time
from scipy.io import wavfile
from pathlib import Path

MODEL        = "distil-large-v3"
DEVICE       = "cuda"
model   = WhisperModel(MODEL, device=DEVICE, compute_type="float32")

start_time = time.time()
# segments, _ = model.transcribe("audio.wav", beam_size=5, language="en")
segments, _ = model.transcribe("audio.wav", language="en", beam_size=5)

import soundfile as sf, numpy as np
data, sr = sf.read("clean.wav")
print("sr=", sr, "dtype=", data.dtype, "NaNs?", np.isnan(data).any())
print("peak=", np.max(np.abs(data)))


printed = ""
for seg in segments:
    print(f"[{seg.start:.2f} - {seg.end:.2f}] {seg.text}")
    if seg.text.startswith(printed):
        new_part = seg.text[len(printed):]
    else: # dropped context, print full
        new_part = seg.text
    if new_part.strip():
        print(new_part, end="", flush=True)
        printed += new_part

print("Transcription completed in:", (time.time() - start_time))
