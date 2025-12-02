import csv
import os
from time import time
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
import message_filters
from prob_rob_msgs.msg import Point2DArrayStamped, Point2D

from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose

import numpy as np


def quat_to_yaw(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(sin_yaw, cos_yaw)

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Lab5Assign3(Node):

    def __init__(self):
        super().__init__('lab5_assign3')
        self.log = self.get_logger()

        self.declare_parameter('landmark_target', 'cyan')
        self.landmark_target = self.get_parameter('landmark_target').get_parameter_value().string_value

        self.log.info(f"Landmark target color: {self.landmark_target}")

        self.link_name = {
            'red':    'landmark_1::link',
            'green':  'landmark_2::link',
            'yellow':   'landmark_3::link',
            'magenta': 'landmark_4::link',
            'cyan':     'landmark_5::link',
        }

        if self.landmark_target not in self.link_name:
            self.log.error(f"Unknown landmark_target '{self.landmark_target}'. Valid options: cyan, yellow, green, magenta, red.")
            raise RuntimeError("Invalid landmark_target parameter.")

        self.landmark_link_name = self.link_name[self.landmark_target]
        self.camera_link_name = 'waffle_pi::camera_link'

        self.log.info(f"Using Gazebo link '{self.landmark_link_name}' "f"for color '{self.landmark_target}'")

        self.name_to_idx = None
        self.cam_idx = None
        self.lmk_idx = None
        self.cam_pose: Pose = None
        self.lmk_pose: Pose = None

        self.meas_topic = f'/estimated_pose/{self.landmark_target}' 
        self.meas_sub = self.create_subscription(Point2DArrayStamped, self.meas_topic, self.measured_callback, 10)

        self.link_states_sub = self.create_subscription(LinkStates,'/gazebo/link_states',self.link_states_callback, 10)

        self.error_topic = f'/meas_error/{self.landmark_target}'
        self.error_pub = self.create_publisher(Point2DArrayStamped, self.error_topic, 10)

        pkg_share = get_package_share_directory("prob_rob_labs")

        # Compute the *source* misc folder manually
        # We look for "src/prob_rob_labs_ros_2"
        ws_src_root = os.path.expanduser("~/ros2_ws/src/prob_rob_labs_ros_2")

        misc_folder = os.path.join(ws_src_root, "misc")
        os.makedirs(misc_folder, exist_ok=True)   # ensure folder exists

        self.csv_path = os.path.join(
            misc_folder,
            f"measurement_errors_{self.landmark_target}.csv"
        )

        self.get_logger().info(f"CSV will be saved to: {self.csv_path}")

        # Create header if file does not exist
        if not os.path.exists(self.csv_path):
            self.get_logger().info(f"Creating new CSV log at {self.csv_path}")
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "unix_time",
                    "landmark",
                    "meas_d", "meas_theta",
                    "true_d", "true_theta",
                    "err_d", "err_theta"
                ])
        else:
            self.get_logger().info(f"Appending to existing CSV log: {self.csv_path}")

    def link_states_callback(self, msg: LinkStates):
        if self.name_to_idx is None:
            self.name_to_idx = {name: i for i, name in enumerate(msg.name)}

        if self.camera_link_name not in self.name_to_idx:
                self.log.error(f"Camera link '{self.camera_link_name}' not found in /gazebo/link_states.")
                return

        if self.landmark_link_name not in self.name_to_idx:
                self.log.error(f"Landmark link '{self.landmark_link_name}' not found in /gazebo/link_states.")
                return

        self.cam_idx = self.name_to_idx[self.camera_link_name]
        self.lmk_idx = self.name_to_idx[self.landmark_link_name]

        try:
                self.cam_pose = msg.pose[self.cam_idx]
                self.lmk_pose = msg.pose[self.lmk_idx]
        except (IndexError, TypeError) as e:
                self.get_logger().warn(f"Failed to cache poses from LinkStates: {e}")
                return
        
    def measured_callback(self, msg: Point2DArrayStamped):
        if self.cam_pose is None or self.lmk_pose is None:
            self.get_logger().warn(
                "No ground truth poses cached yet (camera or landmark). Skipping measurement."
            )
            return
         
        if not msg.points:
            self.get_logger().warn("Received measurement with no points.")
            return
         
        d_meas = float(msg.points[0].x)
        theta_meas = float(msg.points[0].y)

        if not np.isfinite(d_meas) or not np.isfinite(theta_meas):
            self.log.warn("Non-finite measurement received. Skipping.")
            return

        d_true, theta_true = self.compute_true(self.cam_pose, self.lmk_pose)

        e_d = d_meas - d_true
        e_theta = normalize_angle(theta_meas - theta_true)

        self.log.info(
            f"[{self.landmark_target}] d_meas={d_meas:.3f}, d_true={d_true:.3f}, "
            f"θ_meas={theta_meas:.3f}, θ_true={theta_true:.3f}, "
            f"error_d={e_d:.3f}, error_θ={e_theta:.3f}"
        )

        err_msg = Point2DArrayStamped()
        err_msg.header.stamp = self.get_clock().now().to_msg()
        err_msg.header.frame_id = 'camera_link'

        pt = Point2D()
        pt.x = e_d
        pt.y = e_theta
        err_msg.points = [pt]

        self.error_pub.publish(err_msg)

        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time(),                   # unix_time
                    self.landmark_target,     # landmark
                    d_meas,
                    theta_meas,
                    d_true,
                    theta_true,
                    e_d,
                    e_theta
                ])
        except Exception as ex:
            self.log.error(f"Failed to write CSV row: {ex}")

    def compute_true(self, cam_pose: Pose, lmk_pose: Pose):
        cx, cy = cam_pose.position.x, cam_pose.position.y
        lx, ly = lmk_pose.position.x, lmk_pose.position.y

        dx = lx - cx
        dy = ly - cy

        d_true = float(np.hypot(dx, dy))

        phi_world = np.arctan2(dy, dx)

        yaw_cam = quat_to_yaw(cam_pose.orientation)

        theta_true = normalize_angle(phi_world - yaw_cam)

        return d_true, theta_true

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    lab5_assign3 = Lab5Assign3()
    try:
        lab5_assign3.spin()
    finally:
        lab5_assign3.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
