from natural_language_processing.speech_to_text.realtime_whisper import RealtimeSpeechToTextModel
from skills_manager.ros_utils import SpinningRosNode
from std_msgs.msg import String
import rclpy

class RealtimeSpeechToTextModelNode(RealtimeSpeechToTextModel, SpinningRosNode):
    def __init__(self):
        super(RealtimeSpeechToTextModelNode, self).__init__()
        self.pub = self.create_publisher(String, "/nlp/whisper", 5)

    def publish_text(self, text):
        self.pub.publish(String(data=text))

def main():
    rclpy.init()
    rsst = RealtimeSpeechToTextModelNode()
    rsst.node()

if __name__ == "__main__":
    main()