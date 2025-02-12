
import rclpy
import json
import numpy as np
from rclpy.node import Node
from hri_msgs.msg import HRICommand as HRICommandMSG # Download https://github.com/ichores-research/modality_merging to workspace
from pynput import keyboard

from natural_language_processing.speech_to_text.audio_recorder import AudioRecorder
from natural_language_processing.speech_to_text.whisperx_model import SpeechToTextModel
from natural_language_processing.sentence_instruct_transformer.sentence_processor import SentenceProcessor
from natural_language_processing.scene_reader import attach_all_labels

RECORD_TIME = 5
RECORD_NAME = "recording.wav"

class NLInputPipePublisher(Node):
    def __init__(self):
        super(NLInputPipePublisher, self).__init__("nlinput_node")
        self.user = self.declare_parameter("user_name", "casper").get_parameter_value().string_value # replaced if launch
        
        self.pub_original = self.create_publisher(HRICommandMSG, "/modality/nlp_original", 5)
        self.pub = self.create_publisher(HRICommandMSG, "/modality/nlp", 5)

        self.stt = SpeechToTextModel()
        self.sentence_processor = SentenceProcessor()

        self.rec = AudioRecorder()

    def forward(self, recording_name: str):
        print("1. Speech to text", flush=True)
        sentence_text = self.stt.forward(recording_name)
        print("Sentence text: ", sentence_text, flush=True)
        print("2. Sentence processing", flush=True)
        output = self.sentence_processor.predict(sentence_text)

        # output = attach_all_labels(output)

        for k in output.keys():
            if isinstance(output[k], np.ndarray):
                output[k] = list(output[k])

        print("sending this command", flush=True)
        print(output, flush=True)

        self.pub_original.publish(HRICommandMSG(data=[str(json.dumps(output))]))

    # Keyboard event listener
    def on_press(self, key):
        try:
            if key == keyboard.Key.space:  # Start recording on space key press
                self.rec.start_recording()#, sound_card=1)
            if key == keyboard.Key.esc:
                return False
            if key == keyboard.Key.ctrl:
                # test NLP from pre-prepared sound files
                print("Test processing started")
                test_lang = "test_nlp_sentences/test_language.wav"
                test_point = "test_nlp_sentences/test_pointing.wav"
                test_lang_ref = "test_nlp_sentences/test_language_ref.wav"
                test_point_ref = "test_nlp_sentences/test_pointing_ref.wav"
                # self.forward(test_lang)
                # self.forward(test_point)
                # self.forward(test_lang_ref)
                self.forward(test_point_ref)
        except AttributeError:
            pass

    def on_release(self, key):
        if key == keyboard.Key.space:  # Stop recording on space key release
            recording_name, _ = self.rec.stop_recording()
            if recording_name is not None:
                print("Processing started", flush=True)
                self.forward(recording_name)
                
        if key == keyboard.Key.esc:  # Exit on ESC key release
            return False


def main():
    rclpy.init()
    nl_input = NLInputPipePublisher()
    # Listen to keyboard events
    with keyboard.Listener(on_press=nl_input.on_press, on_release=nl_input.on_release) as listener:
        print(f"Press 'space' to start {RECORD_TIME} second recording... Press 'esc' to exit.", flush=True)
        listener.join()

if __name__ == "__main__":
    main()

