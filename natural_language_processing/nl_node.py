
import rclpy
import json
from rclpy.node import Node
from hri_msgs.msg import HRICommand as HRICommandMSG # Download https://github.com/ichores-research/modality_merging to workspace
from pynput import keyboard

from natural_language_processing.speech_to_text.audio_recorder import start_recording, stop_recording
from natural_language_processing.speech_to_text.whisper_model import TextToSpeechModel
from natural_language_processing.sentence_instruct_transformer.sentence_processor import SentenceProcessor
from natural_language_processing.scene_reader import attach_all_labels

class NLInputPipePublisher(Node):
    def __init__(self):
        super(NLInputPipePublisher, self).__init__("nlinput_node")
        self.pub = self.create_publisher(HRICommandMSG, "/modality/nlp", 5)

        self.stt = TextToSpeechModel()
        self.sentence_processor = SentenceProcessor()

    def forward(self, recording_name: str):
        print("1. Speech to text")
        sentence_text = self.stt.forward(recording_name)
        print("2. Sentence processing")
        output = self.sentence_processor.predict(sentence_text)

        output = attach_all_labels(output)

        self.pub.publish(HRICommandMSG(data=str(json.dump(output))))
                         
    # Keyboard event listener
    def on_press(self, key):
        try:
            if key == keyboard.Key.space:  # Start recording on space key press
                start_recording()
        except AttributeError:
            pass

    def on_release(self, key):
        if key == keyboard.Key.space:  # Stop recording on space key release
            recording_name = stop_recording()
            if recording_name is not None:
                print("Processing started")
                self.forward(recording_name)
                
        if key == keyboard.Key.esc:  # Exit on ESC key release
            return False

def main():
    rclpy.init()
    nl_input = NLInputPipePublisher()
    # Listen to keyboard events
    with keyboard.Listener(on_press=nl_input.on_press, on_release=nl_input.on_release) as listener:
        print("Press 'space' to start/stop recording... Press 'esc' to exit.")
        listener.join()

if __name__ == "__main__":
    main()

