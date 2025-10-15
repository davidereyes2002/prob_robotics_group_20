import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseStamped, TwistStamped

heartbeat_period = 0.1

class GroundTruthPublisher(Node):

    def __init__(self):
        super().__init__('ground_truth_publisher')

        self.declare_parameter('frame_id', 'odom')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.link_name = 'waffle_pi::base_footprint'

        self.sub = self.create_subscription(LinkStates, '/gazebo/link_states', self.link_callback, 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/tb3/ground_truth/pose', 10)
        self.pub_twist = self.create_publisher(TwistStamped, '/tb3/ground_truth/twist', 10)

        self.timer = self.create_timer(heartbeat_period, self.publish_ground_truth)

        self.latest_pose = None
        self.latest_twist = None

        self.get_logger().info(f"Ground truth node started. Tracking: {self.link_name}, frame_id: {self.frame_id}")


    def link_callback(self, msg: LinkStates):
        try:
            idx = msg.name.index(self.link_name)
        except ValueError:
            return

        self.latest_pose = msg.pose[idx]
        self.latest_twist = msg.twist[idx]

    def publish_ground_truth(self):
        if self.latest_pose is None or self.latest_twist is None:
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose = self.latest_pose

        twist_msg = TwistStamped()
        twist_msg.header = pose_msg.header
        twist_msg.twist = self.latest_twist

        self.pub_pose.publish(pose_msg)
        self.pub_twist.publish(twist_msg)

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    ground_truth_publisher = GroundTruthPublisher()
    ground_truth_publisher.spin()
    ground_truth_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
