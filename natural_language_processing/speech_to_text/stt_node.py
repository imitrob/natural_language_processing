"""Speech-to-text server node.

Manual mode keeps the original file service. Auto mode additionally starts
the microphone listener from wake_word_listener.py. Both modes expose the
PCM service.

Both services answer with the plain transcription and, in words_json, one entry
per word carrying its real start/end and the model's alternatives for it; the
text is built from that same stream, so the two can never disagree.
"""
import argparse
import json
import os
import threading

import numpy as np
import rclpy
from rclpy.node import Node

from hri_msgs.srv import Transcribe, TranscribeAudio
from natural_language_processing.speech_to_text.stt_client import STT_SERVICE

STT_AUDIO_SERVICE = "/speech_to_text/transcribe_audio"


def fill_response(response, words):
    """Text plus words_json, both built from the one word stream."""
    response.text = " ".join(word["word"] for word in words)
    response.words_json = json.dumps(words)
    return response


class SpeechToTextNode(Node):
    def __init__(self, model=None):
        super().__init__("speech_to_text_node")
        if model is None:
            from natural_language_processing.speech_to_text.whisper_model import SpeechToTextModel
            model = SpeechToTextModel(device="cuda")
        self.model = model  # any object with __call__(file) -> text
        self._model_lock = threading.Lock()
        self.create_service(Transcribe, STT_SERVICE, self.transcribe_callback)
        self.create_service(TranscribeAudio, STT_AUDIO_SERVICE,
                            self.transcribe_audio_callback)

    def transcribe_callback(self, request, response):
        print(f"Transcribing: {request.file}", flush=True)
        # An unreadable file transcribes to "", which the caller cannot tell
        # apart from silence -- so say which file, and that it is not silence.
        if not os.path.isfile(request.file):
            print(f"No such recording: {request.file} (the recorder and this "
                  f"server must share a filesystem, and the path must be "
                  f"absolute), returning empty text", flush=True)
            response.text = ""
            response.words_json = ""
            return response
        try:
            with self._model_lock:
                fill_response(response, self.model.transcribe_to_words(
                    request.file, stamp=request.stamp))
        except Exception as e:  # noqa: BLE001 -- one bad request must not kill the server
            print(f"Transcription failed ({e}), returning empty text", flush=True)
            response.text = ""
            response.words_json = ""
        return response

    def transcribe_audio_words(self, audio, sample_rate, stamp: float = 0.0):
        """Word dicts for signed 16-bit PCM."""
        with self._model_lock:
            return self.model.transcribe_audio_words(audio, sample_rate, stamp=stamp)

    def transcribe_audio_callback(self, request, response):
        try:
            audio = np.asarray(request.audio, dtype=np.int16)
            fill_response(response, self.transcribe_audio_words(
                audio, request.sample_rate, request.stamp))
        except Exception as e:  # noqa: BLE001 -- one bad request must not kill the server
            print(f"PCM transcription failed ({e}), returning empty text", flush=True)
            response.text = ""
            response.words_json = ""
        return response


def main(argv=None):
    parser = argparse.ArgumentParser(description="Speech-to-text server")
    parser.add_argument("--interaction", choices=("manual", "auto"), default="auto")
    parser.add_argument("--audio-device", help="PortAudio input index or device-name substring")
    args, _ = parser.parse_known_args(argv)

    rclpy.init()
    node = None
    try:
        if args.interaction == "auto":
            from natural_language_processing.speech_to_text.wake_word_listener import AutoSpeechToTextNode
            node = AutoSpeechToTextNode(audio_device=args.audio_device)
            node.start_listening()
        else:
            node = SpeechToTextNode()
        print(f"Speech-to-text services ready on {STT_SERVICE} and {STT_AUDIO_SERVICE}",
              flush=True)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if args.interaction == "auto" and node is not None:
            node.stop_listening()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
