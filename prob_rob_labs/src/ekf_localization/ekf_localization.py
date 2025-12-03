import rclpy
from rclpy.node import Node
import yaml
import os
import numpy as np
import math

from sensor_msgs.msg import CameraInfo
from prob_rob_msgs.msg import Point2DArrayStamped, Point2D

Min_pts = 4
Min_x_margin_percentage = 0.02
Min_y_margin_percentage = 0.03
y_max_percentage = 0.625
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster

EPS_ANG = 1e-6
CAM_OFFSET = np.array([0.076, 0, 0.093])

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

class EkfLocalization(Node):

    def __init__(self):
        super().__init__('ekf_localization')
        self.log = self.get_logger()

        self.declare_parameter('map_file', '')
        self.declare_parameter('alpha_v', 0.02)
        self.declare_parameter('alpha_w', 0.02) 
        self.declare_parameter('meas_sigma_d_a', 0.017340)
        self.declare_parameter('meas_sigma_d_b', 0.001990)
        self.declare_parameter('meas_sigma_t_a', 0.000104)
        self.declare_parameter('meas_sigma_t_b', 0.000088)

        map_file = self.get_parameter('map_file').value

        if map_file == '':
            self.log.error("Pass full path of the map_file via launch.")
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

        self.p_matrix = None
        self.img_w = None
        self.img_h = None

        self.camera_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.measurement_pubs = {}
        for color, info in self.landmarks["landmarks"].items():
            meas_topic = f"/measurement_{color}"
            self.measurement_pubs[color] = self.create_publisher(
                Point2DArrayStamped, meas_topic, 10
            )

            vision_topic = f"/vision_{color}/corners"
            self.create_subscription(
                Point2DArrayStamped,
                vision_topic,
                lambda msg, color=color, info=info: self.process_corner_message(msg, color, info),
                10
            )

            self.log.info(f"Subscribed to {vision_topic} and publishing to {meas_topic}")

        self.initial_x = -1.5
        self.initial_y = 0.0
        self.initial_theta = 0.0

        self.x = np.array([[self.initial_theta], [self.initial_x], [self.initial_y]], dtype=float)
        self.P = np.eye(3) * 1e-3
        self.system_time = None
        self.last_vel = 0.0
        self.last_omega = 0.0

        self.create_subscription(Odometry, '/ekf_odom', self.odom_cb, 10)
        self.ekf_pub = self.create_publisher(Odometry, '/ekf_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.alpha_v = self.get_parameter('alpha_v').value
        self.alpha_w = self.get_parameter('alpha_w').value
        self.meas_sigma_d_a = self.get_parameter('meas_sigma_d_a').value
        self.meas_sigma_d_b = self.get_parameter('meas_sigma_d_b').value
        self.meas_sigma_t_a = self.get_parameter('meas_sigma_t_a').value
        self.meas_sigma_t_b = self.get_parameter('meas_sigma_t_b').value

        self.log.info("EKF Localization Node Initialized\n")

    def camera_info_callback(self, msg: CameraInfo):
        if self.p_matrix is None:
            self.p_matrix = np.array(msg.p).reshape(3, 4)[:, :3]
            self.img_w = msg.width
            self.img_h = msg.height

            self.log.info("CameraInfo received and stored.")

        if self.camera_sub:
            self.destroy_subscription(self.camera_sub)
            self.camera_sub = None

    def process_corner_message(self, corners_msg, color, landmark_info):

        if self.p_matrix is None:
            return

        height = landmark_info["height"]

        dist, bearing = self.estimate_distance_bearing(corners_msg, height)

        if dist is None or bearing is None:
            return

        t_meas = self.get_clock().now().nanoseconds * 1e-9
        if self.system_time is None:
            self.system_time = t_meas
            return

        dt = t_meas - self.system_time
        self.log.warn(f"DEBUG: t_meas={t_meas:.6f}, system_time={self.system_time:.6f}, dt={dt:.6f}")
        if dt < -0.02:
            self.log.warn(f"Late measurement (dt={dt:.3f}s). Ignored.")
            return

        dt = max(dt, 0.0)

        if dt > 0:
            self.ekf_predict(dt)

        self.system_time = t_meas

        self.ekf_update(dist, bearing, landmark_info)

        msg = Point2DArrayStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_rgb_frame"

        pt = Point2D()
        pt.x = float(dist)
        pt.y = float(bearing)
        msg.points = [pt]

        self.measurement_pubs[color].publish(msg)
        self.log.info(
            f"[{color}] Published: distance={dist:.3f} | bearing={bearing:.3f}"
        )

    def estimate_distance_bearing(self, corners_msg, landmark_height):

        points = corners_msg.points
        if not points or len(points) < Min_pts:
            return None, None

        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])

        x_min, x_20, x_80, x_max = float(np.min(x)), float(np.percentile(x, 20)), float(np.percentile(x, 80)), float(np.max(x))
        y_min, y_20, y_80, y_max = float(np.min(y)), float(np.percentile(y, 20)), float(np.percentile(y, 80)), float(np.max(y))

        x_margin = min([x_min, self.img_w - x_max])
        y_margin = min([y_min, self.img_h - y_max])

        if x_margin < self.img_w * Min_x_margin_percentage:
            return None, None

        if (y_max > self.img_h * y_max_percentage) or (y_margin < self.img_h * Min_y_margin_percentage):
            return None, None

        x_center_perceived = (x_80 + x_20) / 2.0
        height_perceived = y_max - y_min

        cx = self.p_matrix[0][2]
        fx = self.p_matrix[0][0]
        fy = self.p_matrix[1][1]

        bearing = np.arctan((cx - x_center_perceived) / fx)
        distance = landmark_height * fy / (height_perceived * np.cos(bearing))
        return distance, bearing

    def odom_cb(self, odom_msg):
        t_odom = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec*1e-9

        if self.system_time is None:
            self.last_vel = odom_msg.twist.twist.linear.x
            self.last_omega = odom_msg.twist.twist.angular.z
            return

        dt = t_odom - self.system_time
        if dt < 0:
            self.log.warn("Late odometry sample ignored")
            return
        if dt == 0:
            self.last_vel = odom_msg.twist.twist.linear.x
            self.last_omega = odom_msg.twist.twist.angular.z
            return

        self.last_vel = odom_msg.twist.twist.linear.x
        self.last_omega = odom_msg.twist.twist.angular.z

        self.ekf_predict(dt)
        self.system_time = t_odom
        self.publish_ekf_pose()

    def ekf_predict(self, dt):
        theta = float(self.x[0, 0])
        v = float(self.last_vel)
        omega = float(self.last_omega)

        if abs(omega) < EPS_ANG:
            dtheta = 0.0
            dx = v * dt * math.cos(theta)
            dy = v * dt * math.sin(theta)
        else:
            dtheta = omega * dt
            dx = (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            dy = (v / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))

        theta_new = wrap_angle(theta + dtheta)
        self.x[0, 0] = theta_new
        self.x[1, 0] += dx
        self.x[2, 0] += dy

        G = np.eye(3)
        if abs(omega) < EPS_ANG:
            G[1, 0] = -v * dt * math.sin(theta)
            G[2, 0] = v * dt * math.cos(theta)
        else:
            G[1, 0] = -(v / omega) * math.cos(theta) + (v / omega) * math.cos(theta + omega * dt)
            G[2, 0] = -(v / omega) * math.sin(theta) + (v / omega) * math.sin(theta + omega * dt)

        M = np.diag([self.alpha_v * (v ** 2 + 1e-6), self.alpha_w * (omega ** 2 + 1e-6)])

        V = np.zeros((3, 2), dtype=float)
        if abs(omega) < EPS_ANG:
            V[1, 0] = dt * math.cos(theta)
            V[2, 0] = dt * math.sin(theta)

            V[0, 1] = dt
            V[1, 1] = v * ( -0.5 * (dt ** 2) * math.sin(theta) )
            V[2, 1] = v * ( 0.5 * (dt ** 2) * math.cos(theta) )
        else:
            V[1, 0] = (1.0 / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            V[2, 0] = (1.0 / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))

            V[0, 1] = dt
            V[1, 1] = v * ( (dt * math.cos(theta + omega * dt) * omega - math.sin(theta + omega * dt) + math.sin(theta)) / (omega ** 2) )
            V[2, 1] = v * ( (dt * math.sin(theta + omega * dt) * omega + math.cos(theta + omega * dt) - math.cos(theta)) / (omega ** 2) )
            
        self.P = G @ self.P @ G.T + V @ M @ V.T

    def ekf_update(self, dist, bearing, landmark_info):
        theta = float(self.x[0,0])
        x_r   = float(self.x[1,0])
        y_r   = float(self.x[2,0])

        dx = CAM_OFFSET[0]
        dy = CAM_OFFSET[1]

        x_cam = x_r + math.cos(theta)*dx - math.sin(theta)*dy
        y_cam = y_r + math.sin(theta)*dx + math.cos(theta)*dy

        lm_x = landmark_info["x"]
        lm_y = landmark_info["y"]

        dx_l = lm_x - x_cam
        dy_l = lm_y - y_cam

        pred_dist = math.hypot(dx_l, dy_l)
        pred_bearing = math.atan2(dy_l, dx_l) - theta
        pred_bearing = wrap_angle(pred_bearing)

        q = dx_l**2 + dy_l**2
        sqrt_q = math.sqrt(q)

        H = np.zeros((2,3))

        H[0,1] = -dx_l / sqrt_q
        H[0,2] = -dy_l / sqrt_q

        H[1,0] = -1
        H[1,1] =  dy_l / q
        H[1,2] = -dx_l / q

        sigma_d2 = self.meas_sigma_d_a + self.meas_sigma_d_b * dist
        sigma_t2 = self.meas_sigma_t_a + self.meas_sigma_t_b * dist
        R = np.diag([sigma_d2, sigma_t2])

        z = np.array([[dist],[bearing]])
        z_pred = np.array([[pred_dist],[pred_bearing]])

        y = z - z_pred
        y[1,0] = wrap_angle(y[1,0])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        state_cam = np.array([[theta],[x_cam],[y_cam]]) + K @ y
        state_cam[0,0] = wrap_angle(state_cam[0,0])

        θ_new = state_cam[0,0]

        x_new = state_cam[1,0] - math.cos(θ_new)*dx + math.sin(θ_new)*dy
        y_new = state_cam[2,0] - math.sin(θ_new)*dx - math.cos(θ_new)*dy

        self.x[0,0] = θ_new
        self.x[1,0] = x_new
        self.x[2,0] = y_new

        self.P = (np.eye(3) - K @ H) @ self.P

        self.publish_ekf_pose()

    def publish_ekf_pose(self):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        theta, x, y = self.x.flatten()
        quat = quaternion_from_euler(0,0,theta)

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0] = self.P[1, 1]
        pose_cov[0, 1] = self.P[1, 2]
        pose_cov[0, 5] = self.P[1, 0]
        pose_cov[1, 0] = self.P[2, 1]
        pose_cov[1, 1] = self.P[2, 2]
        pose_cov[1, 5] = self.P[2, 0]
        pose_cov[5, 0] = self.P[0, 1]
        pose_cov[5, 1] = self.P[0, 2]
        pose_cov[5, 5] = self.P[0, 0]
        odom.pose.covariance = pose_cov.flatten().tolist()

        self.ekf_pub.publish(odom)

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
