from natural_language_processing.speech_to_text.realtime_whisper import RealtimeSpeechToTextModel
from skills_manager.ros_utils import SpinningRosNode
from hri_msgs.msg import WhisperText
from std_msgs.msg import Header
import rclpy

class RealtimeSpeechToTextModelNode(RealtimeSpeechToTextModel, SpinningRosNode):
    def __init__(self):
        super(RealtimeSpeechToTextModelNode, self).__init__()
        self.pub = self.create_publisher(WhisperText, "/nlp/whisper", 5)

    def publish_text(self, 
                     new_text: str, # newly transcribed text
                     all_text: str, # all text transcribed from the start
                     ):
        self.pub.publish(WhisperText(header=Header(stamp = self.get_clock().now().to_msg()),
                                     new_text=new_text, all_text=all_text))

def main():
    rclpy.init()
    rsst = RealtimeSpeechToTextModelNode()
    rsst.node()

if __name__ == "__main__":
    main()