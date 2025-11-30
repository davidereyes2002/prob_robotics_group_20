import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler

# heartbeat_period = 0.1

class EKFNode(Node):

    def __init__(self):
        super().__init__('EKFNode')

        self.odom_pub = self.create_publisher(Odometry, '/ekf_odom', 10)
        self.imu_sub = Subscriber(self, Imu, '/imu')
        self.joint_sub = Subscriber(self, JointState, '/joint_states')

        self.sync = ApproximateTimeSynchronizer(
            [self.imu_sub, self.joint_sub],
            queue_size=20,
            slop=0.05
        )
        self.sync.registerCallback(self.sensor_callback)

        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.current_cmd_vel = Twist()

        self.r_w = 0.033
        self.R_bot = 0.1435 / 2
        self.for_factor_linear = 0.921
        self.for_factor_angular = 0.749
        self.input_gain = 0.9

        self.dt_threshold = 1.

        self.initial_x = -1.5
        self.initial_y = 0.0
        self.initial_theta = 0.0

        self.x = np.array([
            [self.initial_theta],
            [self.initial_x],
            [self.initial_y],
            [0.0],
            [0.0]
        ])

        self.state_covariance = np.zeros((5, 5))
        self.last_time = None
        
        self.input_covariance = np.diag([0.02, 0.02])
        self.G_u = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [self.input_gain * (1 - self.for_factor_linear), 0],
            [0, self.input_gain * (1 - self.for_factor_angular)]
        ])
        self.Q = self.G_u @ self.input_covariance @ self.G_u.T

        self.H = np.array([
            [0, 0, 0, 1 / self.r_w,  self.R_bot / self.r_w],
            [0, 0, 0, 1 / self.r_w, -self.R_bot / self.r_w],
            [0, 0, 0, 0, 1]
        ])

        self.imu_covariance_scalar = 0.01
        self.R = np.diag([0.05, 0.05, 0.01])

        self.log = self.get_logger()
        self.log.info('EKF Node initialized.')

    def cmd_vel_callback(self, msg: Twist):
        self.current_cmd_vel = msg
    
    def state_jacobian(self, x, dt):
        theta, _, _, v, _ = x.flatten()

        F = np.eye(5)
        F[0, 4] = dt
        F[1, 0] = -v * dt * math.sin(theta)
        F[1, 3] = dt * math.cos(theta)
        F[2, 0] = v * dt * math.cos(theta)
        F[2, 3] = dt * math.sin(theta)
        F[3, 3] = self.for_factor_linear
        F[4, 4] = self.for_factor_angular

        return F

    def predict_next_state(self, x_prev, v_cmd, omega_cmd, dt):
        next_state = np.zeros((5, 1))

        theta, x, y, v, omega = x_prev.flatten()

        next_state[0] = theta + omega * dt
        next_state[1] = x + v * dt * math.cos(theta)
        next_state[2] = y + v * dt * math.sin(theta)
        next_state[3] = self.for_factor_linear * v + self.input_gain * (1 - self.for_factor_linear) * v_cmd
        next_state[4] = self.for_factor_angular * omega + self.input_gain * (1 - self.for_factor_angular) * omega_cmd

        return next_state

    def sensor_callback(self, imu_msg, joint_msg):
        t = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t

        if dt > self.dt_threshold:
            return

        omega_g = imu_msg.angular_velocity.z
        imu_var = imu_msg.angular_velocity_covariance[8] if imu_msg.angular_velocity_covariance[8] != 0 else self.imu_covariance_scalar

        try:
            idx_l = joint_msg.name.index('wheel_left_joint')
            idx_r = joint_msg.name.index('wheel_right_joint')
            omega_l = joint_msg.velocity[idx_l]
            omega_r = joint_msg.velocity[idx_r]
        except ValueError:
            self.log.warn("Wheel joint names not found in JointState")
            return
        
        z = np.array([[omega_r], [omega_l], [omega_g]])
        self.R = np.diag([0.05, 0.05, imu_var])

        v_cmd = self.current_cmd_vel.linear.x
        omega_cmd = self.current_cmd_vel.angular.z
        
        x_pred = self.predict_next_state(self.x, v_cmd, omega_cmd, dt)
        F = self.state_jacobian(self.x, dt)
        state_covariance_pred = F @ self.state_covariance @ F.T + self.Q

        S = self.H @ state_covariance_pred @ self.H.T + self.R
        K = state_covariance_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.state_covariance = (np.eye(5) - K @ self.H) @ state_covariance_pred

        self.log.info(
            f"dt={dt:.3f}s | ω_l={omega_l:.3f} ω_r={omega_r:.3f} ω_g={omega_g:.3f} | "
            f"cmd_vel=({v_cmd:.2f}, {omega_cmd:.2f})"
        )

        self.publish_odometry(imu_msg.header.stamp)

    def publish_odometry(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        theta, x, y, v, omega = self.x.flatten()
        quat = quaternion_from_euler(0, 0, theta)

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0] = self.state_covariance[1, 1]
        pose_cov[0, 1] = self.state_covariance[1, 2]
        pose_cov[0, 5] = self.state_covariance[1, 0]
        pose_cov[1, 0] = self.state_covariance[2, 1]
        pose_cov[1, 1] = self.state_covariance[2, 2]
        pose_cov[1, 5] = self.state_covariance[2, 0]
        pose_cov[5, 0] = self.state_covariance[0, 1]
        pose_cov[5, 1] = self.state_covariance[0, 2]
        pose_cov[5, 5] = self.state_covariance[0, 0]
        odom.pose.covariance = pose_cov.flatten().tolist()

        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = omega

        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = self.state_covariance[3, 3]
        twist_cov[0, 5] = self.state_covariance[3, 4]
        twist_cov[5, 0] = self.state_covariance[4, 3]
        twist_cov[5, 5] = self.state_covariance[4, 4]
        odom.twist.covariance = twist_cov.flatten().tolist()

        self.odom_pub.publish(odom)

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = EKFNode()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
