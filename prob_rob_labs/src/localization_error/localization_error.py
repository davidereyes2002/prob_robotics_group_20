import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion
import math

def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

class LocalizationError(Node):

    def __init__(self):
        super().__init__('localization_error')
        
        self.gt_pose = None
        self.ekf_pose = None

        self.sub_gt = self.create_subscription(
            PoseStamped,
            '/tb3/ground_truth/pose',
            self.gt_callback,
            10
        )

        self.sub_ekf = self.create_subscription(
            Odometry,
            '/ekf_pose',
            self.ekf_callback,
            10
        )

        self.pub_pos_err = self.create_publisher(Float64, '/ekf_error/position', 10)
        self.pub_yaw_err = self.create_publisher(Float64, '/ekf_error/orientation', 10)

        self.get_logger().info("EKF Error Node initialized.")

    def gt_callback(self, msg: PoseStamped):
        self.gt_pose = msg.pose
    
    def ekf_callback(self, msg: Odometry):
        self.ekf_pose = msg.pose.pose
        self.compute_error()

    def compute_error(self):
        if self.gt_pose is None or self.ekf_pose is None:
            return

        gt_x = self.gt_pose.position.x
        gt_y = self.gt_pose.position.y
        ekf_x = self.ekf_pose.position.x
        ekf_y = self.ekf_pose.position.y

        pos_err = math.sqrt((gt_x - ekf_x)**2 + (gt_y - ekf_y)**2)

        q_gt = self.gt_pose.orientation
        q_ekf = self.ekf_pose.orientation

        yaw_gt = euler_from_quaternion([q_gt.x, q_gt.y, q_gt.z, q_gt.w])[2]
        yaw_ekf = euler_from_quaternion([q_ekf.x, q_ekf.y, q_ekf.z, q_ekf.w])[2]

        yaw_err = wrap_angle(yaw_gt - yaw_ekf)

        msg_p = Float64()
        msg_p.data = pos_err
        self.pub_pos_err.publish(msg_p)

        msg_y = Float64()
        msg_y.data = abs(yaw_err)
        self.pub_yaw_err.publish(msg_y)

        self.get_logger().info(
            f"PosErr: {pos_err:.3f} m | YawErr: {math.degrees(yaw_err):.2f}°"
        )

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = LocalizationError()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
