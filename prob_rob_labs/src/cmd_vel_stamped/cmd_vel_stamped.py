import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

class CmdVelStamped(Node):
    def __init__(self):
        super().__init__('cmd_vel_stamper')
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.callback, 10)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)

    def callback(self, msg: Twist):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = 'odom'
        stamped.twist = msg
        self.pub.publish(stamped)

def main():
    rclpy.init()
    node = CmdVelStamped()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
