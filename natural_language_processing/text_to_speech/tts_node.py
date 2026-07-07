"""Text-to-speech server node: loads the model once and serves speak
requests (see tts_client.py for the client side). Any model script with the
speak(text) API can be served here (kokoro_model.Chatterbox by default)."""
import rclpy
from rclpy.node import Node

from hri_msgs.srv import Speak
from natural_language_processing.text_to_speech.tts_client import TTS_SERVICE


class TextToSpeechNode(Node):
    def __init__(self, model=None):
        super().__init__("text_to_speech_node")
        if model is None:
            from natural_language_processing.text_to_speech.kokoro_model import Chatterbox
            model = Chatterbox(device="cuda")
        self.model = model  # any object with speak(text)
        self.create_service(Speak, TTS_SERVICE, self.speak_callback)

    def speak_callback(self, request, response):
        print(f"Speaking: {request.text}", flush=True)
        self.model.speak(request.text)
        response.success = True
        return response


def main():
    rclpy.init()
    node = TextToSpeechNode()
    print(f"Text-to-speech service ready on {TTS_SERVICE}", flush=True)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
