"""Speech-to-text server node: loads the model once and serves transcription
requests (see stt_client.py for the client side). Any model script with the
__call__(file) -> text API can be served here (whisper_model.py by default)."""
import os

import rclpy
from rclpy.node import Node

from hri_msgs.srv import Transcribe
from natural_language_processing.speech_to_text.stt_client import STT_SERVICE


class SpeechToTextNode(Node):
    def __init__(self, model=None):
        super().__init__("speech_to_text_node")
        if model is None:
            from natural_language_processing.speech_to_text.whisper_model import SpeechToTextModel
            model = SpeechToTextModel(device="cuda")
        self.model = model  # any object with __call__(file) -> text
        self.create_service(Transcribe, STT_SERVICE, self.transcribe_callback)

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
            response.text = self.model(request.file)
        except Exception as e:  # noqa: BLE001 -- one bad request must not kill the server
            print(f"Transcription failed ({e}), returning empty text", flush=True)
            response.text = ""
        return response


def main():
    rclpy.init()
    node = SpeechToTextNode()
    print(f"Speech-to-text service ready on {STT_SERVICE}", flush=True)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
