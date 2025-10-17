import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion
import math



class EKFErrorAnalysis(Node):

    def __init__(self):
        super().__init__('EKF_error_analysis')

        self.sub_gt = self.create_subscription(PoseStamped, '/tb3/ground_truth/pose', self.gt_callback, 10)
        self.sub_ekf = self.create_subscription(Odometry, '/ekf_odom', self.ekf_callback, 10)

        self.pub_pos_err = self.create_publisher(Float64, '/ekf_error/position', 10)
        self.pub_yaw_err = self.create_publisher(Float64, '/ekf_error/orientation', 10)
        
        self.gt_pose = None
        self.ekf_pose = None

        self.log = self.get_logger()
        self.log.info('EKF Error Analysis Node started.')

    def gt_callback(self, msg):
        self.gt_pose = msg

    def ekf_callback(self, msg):
        self.ekf_pose = msg.pose.pose
        self.compare_errors(msg.header.stamp)

    def compare_errors(self, stamp):
        if self.gt_pose is None or self.ekf_pose is None:
            return

        gt_x = self.gt_pose.pose.position.x
        gt_y = self.gt_pose.pose.position.y

        ekf_x = self.ekf_pose.position.x
        ekf_y = self.ekf_pose.position.y

        pos_err = math.sqrt((gt_x - ekf_x)**2 + (gt_y - ekf_y)**2)

        q_gt = self.gt_pose.pose.orientation
        q_ekf = self.ekf_pose.orientation

        yaw_gt = euler_from_quaternion([q_gt.x, q_gt.y, q_gt.z, q_gt.w])[2]
        yaw_ekf = euler_from_quaternion([q_ekf.x, q_ekf.y, q_ekf.z, q_ekf.w])[2]

        yaw_diff = math.atan2(math.sin(yaw_gt - yaw_ekf), math.cos(yaw_gt - yaw_ekf))

        pos_msg = Float64()
        yaw_msg = Float64()
        pos_msg.data = pos_err
        yaw_msg.data = abs(yaw_diff)

        self.pub_pos_err.publish(pos_msg)
        self.pub_yaw_err.publish(yaw_msg)

        self.log.info(f"PosErr={pos_err:.3f} m | YawErr={math.degrees(yaw_diff):.2f}°")

  
    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = EKFErrorAnalysis()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
