import rclpy
from rclpy.node import Node
import yaml
import os

heartbeat_period = 0.1

class EkfLocalization(Node):

    def __init__(self):
        super().__init__('ekf_localization')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        self.declare_parameter('map_file', '')
        map_file = self.get_parameter('map_file').value

        if map_file == '':
            self.log.error("No map_file parameter provided! Pass full path via launch.")
            self.landmarks = {}
            return
        
        if not os.path.exists(map_file):
            self.log.error(f"Landmark file does not exist at passed path:\n{map_file}")
            self.landmarks = {}
            return

        self.log.info(f"Loading landmark map from: {map_file}")

        with open(map_file, 'r') as f:
            self.landmarks = yaml.safe_load(f)
        self.log.info(f"Loaded landmarks: {self.landmarks}")

    def heartbeat(self):
        self.log.info("heartbeat")

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = EkfLocalization()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
