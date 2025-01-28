import rclpy, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from hri_msgs.msg import HRICommand as HRICommandMSG

class NLPPub(Node):
    def __init__(self):
        super().__init__("NLP_pub_node")
        qos_profile = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(HRICommandMSG, "/modality/nlp", qos_profile)

    def pub_random(self):
        t0 = time.time()
        print(f"point scene object 1:")
        time.sleep(2)
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("point scene object 2:")
        time.sleep(2)
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("point scene object 3:")
        time.sleep(2)
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        self.pub.publish(HRICommandMSG(data=[f"{{'target_obect_stamp': [{t0+5.0}, {t0+10.0}, {t0+15.0}]}}"]))
        print("Sending in:")
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("send!")

def main():
    rclpy.init()
    node = NLPPub()
    node.pub_random()
    rclpy.spin(node)  # Keep the node alive to store the message
    rclpy.shutdown()

if __name__ == "__main__":
    main()