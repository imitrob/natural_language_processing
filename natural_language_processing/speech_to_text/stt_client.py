"""Client to the speech-to-text server node (stt_node.py). The model is
loaded in the server; this class only calls the service, keeping the model
API: __call__(file) -> transcribed text.

The client owns a private node + executor, so calling it from inside another
node's callback (e.g. HRI's single-threaded executor) cannot deadlock."""
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from hri_msgs.srv import Transcribe

STT_SERVICE = "/speech_to_text/transcribe"


class SpeechToTextClient():
    def __init__(self, timeout_sec: float = 60.0):
        self.timeout_sec = timeout_sec
        self._node = rclpy.create_node(f"stt_client_{np.random.randint(100000)}")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._client = self._node.create_client(Transcribe, STT_SERVICE)

    def __call__(self, file: str = "") -> str:
        if not self._client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"Speech-to-text service {STT_SERVICE} unavailable. "
                               f"Run: ros2 run natural_language_processing stt_node")
        future = self._client.call_async(Transcribe.Request(file=file))
        rclpy.spin_until_future_complete(self._node, future, executor=self._executor,
                                         timeout_sec=self.timeout_sec)
        if not future.done():
            raise RuntimeError(f"Speech-to-text did not respond within {self.timeout_sec}s")
        return future.result().text

    def transcribe_to_stamped(self, file: str, stamp: float = 0.0):
        """Transcribed words as [[stamp, word], ...] (same heuristic word
        timing as the model's transcribe_to_stamped)."""
        words = self(file).split(" ")
        return [[stamp + n * 0.2, w] for n, w in enumerate(words)]

    def delete(self):
        self._node.destroy_node()
