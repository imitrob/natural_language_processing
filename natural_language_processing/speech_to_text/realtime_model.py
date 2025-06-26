#!/usr/bin/env python3
import sys
from collections import deque
from typing import Optional
import numpy as np
import sounddevice as sd
import torch
import queue
print("[+] Loading model – first run can take a minute…", flush=True)
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR

MODEL_NAME = "nvidia/stt_en_fastconformer_hybrid_large_streaming_80ms"
asr_model = (
    EncDecRNNTBPEModel
    .from_pretrained(model_name=MODEL_NAME)
    .cuda()
    .eval()
)

WIN_STRIDE = asr_model._cfg.preprocessor.window_stride          # 0.008
DUMMY_WAV  = torch.zeros(1, 800, device='cuda')                 # 50 ms
with torch.no_grad():
    feats, _ = asr_model.preprocessor(
        input_signal=DUMMY_WAV,
        length=torch.tensor([800], device='cuda'),
    )
N_STEPS = feats.shape[2]                                        # e.g. 6
FRAME_LEN_SEC = N_STEPS * WIN_STRIDE                            # 0.048

TARGET_SR          = 16_000
FRAME_LEN_SAMPLES  = 800          # 50 ms @16 k

frame_asr = FrameBatchASR(
    asr_model=asr_model,
    frame_len=FRAME_LEN_SEC,   # seconds!
    total_buffer=0.32,         # 0.32 s = 16 frames context
    batch_size=1,
    pad_to_buffer_len=True,
)

# ────────────────── On-the-fly feature generator ───────────────
class MicFeatureIterator:
    def __init__(self, preproc, device):
        self.preproc, self.device = preproc, device
        self.q = queue.Queue(maxsize=64)          # plenty of room

    # called from PortAudio callback
    def put_audio(self, wav: np.ndarray):
        try:
            self.q.put_nowait(wav)
        except queue.Full:
            self.q.get_nowait()       # drop oldest if we ever fall behind
            self.q.put_nowait(wav)

    # iterator protocol
    def __iter__(self): return self

    def __next__(self):
        wav = self.q.get()            # blocks -- *never* raises StopIteration
        wav_t  = torch.from_numpy(wav).to(self.device).unsqueeze(0)
        wav_len = torch.tensor([wav_t.shape[1]], device=self.device)
        with torch.no_grad():
            feats, _ = self.preproc(input_signal=wav_t, length=wav_len)
        return feats.squeeze(0).cpu().numpy()   # (n_mel, n_steps)

mic_iter = MicFeatureIterator(asr_model.preprocessor, asr_model.device)
frame_asr.frame_bufferer.set_frame_reader(mic_iter)

# ───────────────────────── helpers ─────────────────────────────
def _fix_len(chunk: np.ndarray) -> np.ndarray:
    """Ensure exactly 800 samples by ±1-2 pad / trim."""
    if len(chunk) < FRAME_LEN_SAMPLES:
        return np.pad(chunk, (0, FRAME_LEN_SAMPLES - len(chunk)))
    if len(chunk) > FRAME_LEN_SAMPLES:
        return chunk[:FRAME_LEN_SAMPLES]
    return chunk

# ─────────────────────────── main ──────────────────────────────
def main():
    in_dev = 7                                        # your Jabra input ID
    info   = sd.query_devices(in_dev, "input")
    print(info)
    mic_sr = int(info["default_samplerate"])

    if mic_sr != TARGET_SR:
        import librosa
        print(f"[i] Mic SR {mic_sr} Hz ≠ model {TARGET_SR} Hz → will resample.")

    MIC_FRAME_LEN = int(round(FRAME_LEN_SAMPLES * mic_sr / TARGET_SR))
    MIC_FRAME_HOP = MIC_FRAME_LEN                         # no overlap on read

    ring = deque()                                        # raw-audio ring-buf
    frames_enqueued = 0
    def audio_cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        ring.extend(indata[:, 0])                         # mono

    with sd.InputStream(
        device=in_dev,
        channels=1,
        samplerate=mic_sr,
        dtype="float32",
        blocksize=0,
        callback=audio_cb,
    ):
        print("[✓] Listening… (Ctrl-C to quit)")
        partial: str = ""
        try:
            while True:
                # ---------- capture one 50 ms chunk ----------
                if len(ring) >= MIC_FRAME_LEN:
                    mic_chunk = np.fromiter(
                        (ring.popleft() for _ in range(MIC_FRAME_LEN)),
                        dtype=np.float32,
                    )
                    if mic_sr != TARGET_SR:
                        mic_chunk = librosa.resample(
                            mic_chunk, orig_sr=mic_sr, target_sr=TARGET_SR,
                            fix=True, scale=True,
                        )
                    
                    mic_iter.put_audio(_fix_len(mic_chunk))
                    frames_enqueued += 1                      # one window queued

                    # ---------- decode when ready ---------------
                    if frames_enqueued >= 2:
                        text: Optional[str] = frame_asr.transcribe(
                            tokens_per_chunk=1, delay=0
                        )
                        frames_enqueued -= 2                  # both windows consumed
                        print("text", text)
                        if text and text != partial:
                            partial = text
                            print("\r" + partial, end="", flush=True)
                sd.sleep(1)   # 1 ms – keeps loop responsive
        except KeyboardInterrupt:
            print("\n[+] Stopped.")

if __name__ == "__main__":
    main()
