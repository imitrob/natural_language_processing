"""Client to the text-to-speech server node (tts_node.py). The model is
loaded in the server; this class only calls the service, keeping the model
API: speak(text).

The client owns a private node + executor, so calling it from inside another
node's callback (e.g. HRI's single-threaded executor) cannot deadlock."""
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from hri_msgs.srv import Speak

TTS_SERVICE = "/text_to_speech/speak"


class TextToSpeechClient():
    def __init__(self, timeout_sec: float = 120.0):
        self.timeout_sec = timeout_sec
        self._warmed_up = False
        self._node = rclpy.create_node(f"tts_client_{np.random.randint(100000)}")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._client = self._node.create_client(Speak, TTS_SERVICE)

    def speak(self, text: str):
        """Blocks until the text has been spoken (as the model did). Speech is
        not critical -- when the server is unavailable, warn and continue (the
        caller prints the text anyway). The availability check is
        non-blocking (except a one-time discovery wait on the first call), so
        speak() stays cheap when no server is running."""
        if not self._warmed_up:
            self._warmed_up = True
            self._client.wait_for_service(timeout_sec=2.0)  # allow discovery on first use
        if not self._client.service_is_ready():
            print(f"Text-to-speech service {TTS_SERVICE} unavailable, skipping speech. "
                  f"Run: ros2 run natural_language_processing tts_node", flush=True)
            return
        future = self._client.call_async(Speak.Request(text=text))
        rclpy.spin_until_future_complete(self._node, future, executor=self._executor,
                                         timeout_sec=self.timeout_sec)
        if not future.done():
            print(f"Text-to-speech did not respond within {self.timeout_sec}s", flush=True)

    def delete(self):
        self._node.destroy_node()
