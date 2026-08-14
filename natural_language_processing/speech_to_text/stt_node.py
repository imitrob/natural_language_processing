"""Speech-to-text server node.

Manual mode keeps the original file service. Auto mode additionally starts
the microphone listener from auto_stt.py. Both modes expose the PCM service.
"""
import argparse
import os
import threading

import numpy as np
import rclpy
from rclpy.node import Node

from hri_msgs.srv import Transcribe, TranscribeAudio
from natural_language_processing.speech_to_text.stt_client import STT_SERVICE

STT_AUDIO_SERVICE = "/speech_to_text/transcribe_audio"


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
            return response
        try:
            with self._model_lock:
                response.text = self.model(request.file)
        except Exception as e:  # noqa: BLE001 -- one bad request must not kill the server
            print(f"Transcription failed ({e}), returning empty text", flush=True)
            response.text = ""
        return response

    def transcribe_audio(self, audio, sample_rate):
        with self._model_lock:
            return self.model.transcribe_audio(audio, sample_rate)

    def transcribe_audio_callback(self, request, response):
        try:
            response.text = self.transcribe_audio(
                np.asarray(request.audio, dtype=np.int16), request.sample_rate)
        except Exception as e:  # noqa: BLE001 -- one bad request must not kill the server
            print(f"PCM transcription failed ({e}), returning empty text", flush=True)
            response.text = ""
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
            from natural_language_processing.speech_to_text.auto_stt import AutoSpeechToTextNode
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
