"""Speech-to-text server node: loads the model once and serves transcription
requests (see stt_client.py for the client side). Any model script with the
__call__(file) -> text API can be served here (whisper_model.py by default)."""
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
        response.text = self.model(request.file)
        return response


def main():
    rclpy.init()
    node = SpeechToTextNode()
    print(f"Speech-to-text service ready on {STT_SERVICE}", flush=True)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
