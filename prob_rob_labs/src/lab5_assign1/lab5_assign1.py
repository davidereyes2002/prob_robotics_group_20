import rclpy
from rclpy.node import Node

from prob_rob_msgs.msg import Point2DArrayStamped
import numpy as np


class Lab5Assign1(Node):

    def __init__(self):
        super().__init__('lab5_assign1')
        self.sub = self.create_subscription(Point2DArrayStamped, '/vision_cyan/corners', self.measurements_callback, 10)
    
    def measurements_callback(self, msg: Point2DArrayStamped):
        points = msg.points

        if not points:
            self.get_logger().warn("No corners detected for cyan landmark.")
            return

        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])

        height = np.max(y) - np.min(y)

        x_center = (np.max(x) + np.min(x)) / 2.0

        self.get_logger().info(f"[Cyan Landmark] Height: {height:.2f}px | X-center: {x_center:.2f}px | Points: {len(points)}")

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = Lab5Assign1()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
