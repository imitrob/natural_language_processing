"""Always-listening, wake-qualified speech input for the STT server."""
import math
import os
import queue
import re
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper.vad import VadOptions, get_speech_timestamps

from hri_msgs.msg import WhisperText
from natural_language_processing.speech_to_text.stt_node import SpeechToTextNode

# Tune these for the microphone and room.
WAKE_PHRASE = "hey robot"
SAMPLE_RATE = 16_000
BLOCK_SAMPLES = 512  # Silero's native 32 ms window at 16 kHz
PRE_ROLL_SECONDS = 0.3
END_SILENCE_SECONDS = 0.7
MIN_SPEECH_SECONDS = 0.25
MAX_UTTERANCE_SECONDS = 15.0
VAD_THRESHOLD = 0.5
WHISPER_TOPIC = "/nlp/whisper"


def wake_command(text, wake_phrase=WAKE_PHRASE):
    """Return normalized command words only when text starts with the wake phrase."""
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
    wake_words = re.findall(r"[a-z0-9]+", wake_phrase.lower())
    if words[:len(wake_words)] != wake_words:
        return None
    return " ".join(words[len(wake_words):])


class UtteranceSegmenter:
    """Collect VAD-labelled PCM blocks, preserving audio before speech onset."""

    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.pre_roll_samples = round(PRE_ROLL_SECONDS * sample_rate)
        self.end_silence_samples = round(END_SILENCE_SECONDS * sample_rate)
        self.min_speech_samples = round(MIN_SPEECH_SECONDS * sample_rate)
        self.max_utterance_samples = round(MAX_UTTERANCE_SECONDS * sample_rate)
        self._pre_roll = deque()
        self._pre_roll_size = 0
        self._reset_utterance()

    def _reset_utterance(self):
        self._audio = None
        self._speech_samples = 0
        self._silence_samples = 0
        self._utterance_samples = 0
        self.onset = None

    def _remember(self, block):
        self._pre_roll.append(block)
        self._pre_roll_size += len(block)
        while self._pre_roll_size > self.pre_roll_samples:
            excess = self._pre_roll_size - self.pre_roll_samples
            if excess < len(self._pre_roll[0]):
                self._pre_roll[0] = self._pre_roll[0][excess:]
                self._pre_roll_size -= excess
                break
            self._pre_roll_size -= len(self._pre_roll.popleft())

    def push(self, block, is_speech, captured_at=None):
        """Return ``(utterance, onset, started)``; utterance is None until done."""
        block = np.asarray(block, dtype=np.int16).reshape(-1).copy()
        captured_at = time.time() if captured_at is None else captured_at

        if self._audio is None:
            self._remember(block)
            if not is_speech:
                return None, None, False
            self._audio = list(self._pre_roll)
            self.onset = captured_at - len(block) / self.sample_rate
            self._speech_samples = len(block)
            self._utterance_samples = len(block)
            return None, self.onset, True

        self._audio.append(block)
        self._utterance_samples += len(block)
        if is_speech:
            self._speech_samples += len(block)
            self._silence_samples = 0
        else:
            self._silence_samples += len(block)

        finished = (self._silence_samples >= self.end_silence_samples
                    or self._utterance_samples >= self.max_utterance_samples)
        if not finished:
            return None, None, False

        utterance = (np.concatenate(self._audio)
                     if self._speech_samples >= self.min_speech_samples else None)
        onset = self.onset
        self._pre_roll.clear()
        self._pre_roll_size = 0
        self._reset_utterance()
        return utterance, onset, False


def resolve_audio_device(requested):
    requested = requested or os.environ.get("AUDIO_DEVICE")
    if requested is None:
        return None, sd.query_devices(kind="input")
    try:
        device = int(requested)
    except ValueError:
        matches = [(index, info) for index, info in enumerate(sd.query_devices())
                   if info["max_input_channels"] > 0
                   and requested.casefold() in info["name"].casefold()]
        if not matches:
            raise ValueError(f"No input device contains {requested!r}")
        device, info = matches[0]
        if len(matches) > 1:
            print(f"Multiple inputs match {requested!r}; using the first", flush=True)
        return device, info
    return device, sd.query_devices(device, kind="input")


class AutoSpeechToTextNode(SpeechToTextNode):
    def __init__(self, audio_device=None, model=None):
        self.audio_device, self.audio_device_info = resolve_audio_device(audio_device)
        super().__init__(model=model)
        self.publisher = self.create_publisher(WhisperText, WHISPER_TOPIC, 10)
        self._audio_queue = queue.Queue()
        self._utterance_queue = queue.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._segment_thread = None
        self._transcribe_thread = None
        self._vad_options = VadOptions(
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
        )

    def start_listening(self):
        print(f"Auto interaction listening on input {self.audio_device_info['name']} "
              f"at {SAMPLE_RATE} Hz", flush=True)
        self._segment_thread = threading.Thread(target=self._segment_audio, daemon=True)
        self._transcribe_thread = threading.Thread(target=self._transcribe_utterances,
                                                    daemon=True)
        self._segment_thread.start()
        self._transcribe_thread.start()
        self._stream = sd.InputStream(
            device=self.audio_device,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SAMPLES,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_listening(self):
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._audio_queue.put(None)
        if self._segment_thread is not None:
            self._segment_thread.join()
        self._utterance_queue.put(None)
        if self._transcribe_thread is not None:
            self._transcribe_thread.join()

    def _audio_callback(self, audio, _frames, _time_info, status):
        if status:
            print(f"Audio input warning: {status}", flush=True)
        self._audio_queue.put((audio[:, 0].copy(), time.time()))

    def _is_speech(self, block):
        audio = block.astype(np.float32) / 32768.0
        return bool(get_speech_timestamps(
            audio, self._vad_options, sampling_rate=SAMPLE_RATE))

    def _segment_audio(self):
        segmenter = UtteranceSegmenter()
        while not self._stop.is_set():
            item = self._audio_queue.get()
            if item is None:
                return
            block, captured_at = item
            utterance, onset, started = segmenter.push(
                block, self._is_speech(block), captured_at)
            if started:
                print("Speech detected, listening...", flush=True)
            if utterance is not None:
                self._utterance_queue.put((utterance, onset))

    def _transcribe_utterances(self):
        while not self._stop.is_set():
            item = self._utterance_queue.get()
            if item is None:
                return
            audio, onset = item
            try:
                transcript = self.transcribe_audio(audio, SAMPLE_RATE).strip()
            except Exception as error:  # one bad utterance must not stop listening
                print(f"Transcription failed: {error}", flush=True)
                continue
            if self._stop.is_set():
                return
            command = wake_command(transcript)
            if command is None:
                print(f"Ignored (missing {WAKE_PHRASE!r}): {transcript}", flush=True)
                continue
            if not command:
                print("Wake phrase heard without a command", flush=True)
                continue
            print(f"Accepted: {command}", flush=True)
            message = WhisperText(new_text=command, all_text=command)
            seconds = math.floor(onset)
            message.header.stamp.sec = seconds
            message.header.stamp.nanosec = int((onset - seconds) * 1_000_000_000)
            self.publisher.publish(message)
