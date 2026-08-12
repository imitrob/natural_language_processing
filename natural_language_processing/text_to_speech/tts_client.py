"""Client to the text-to-speech server node (tts_node.py). The model is
loaded in the server; this class only calls the service, keeping the model
API: speak(text).

The client owns a private node + executor, so calling it from inside another
node's callback (e.g. HRI's single-threaded executor) cannot deadlock."""
import threading

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from hri_msgs.srv import Speak

TTS_SERVICE = "/text_to_speech/speak"


class TextToSpeechClient():
    def __init__(self):
        self._warmed_up = False
        self._node = rclpy.create_node(f"tts_client_{np.random.randint(100000)}")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._client = self._node.create_client(Speak, TTS_SERVICE)
        # The responses still have to be collected -- an un-spun client leaks
        # pending requests and makes the server fail to deliver its reply -- but
        # nobody has to wait for them, so a background spin does it.
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        while rclpy.ok():
            try:
                self._executor.spin()
                return
            except Exception as error:  # noqa: BLE001 -- speech must not kill it
                print(f"Text-to-speech client spin failed: {error}", flush=True)

    def speak(self, text: str):
        """Say the text, returning immediately. Speech is an announcement, not a
        step: waiting for it would delay the robot behind an utterance, and a
        "stop" arriving mid-sentence has to act now. Utterances still come out in
        order -- the server speaks them one at a time.

        Speech is not critical -- when the server is unavailable, warn and
        continue (the caller prints the text anyway). The availability check is
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
        future.add_done_callback(self._spoken)

    @staticmethod
    def _spoken(future):
        try:
            future.result()
        except Exception as error:  # noqa: BLE001 -- nothing depends on speech
            print(f"Text-to-speech failed: {error}", flush=True)

    def delete(self):
        self._node.destroy_node()
