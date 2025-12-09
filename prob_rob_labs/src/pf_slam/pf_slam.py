import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from prob_rob_msgs.msg import Point2DArrayStamped, Point2D

import os
import yaml
import numpy as np

min_points = 4
min_x_margin_percent = 0.02
min_y_margin_percent = 0.03
y_max_percent = 0.625

class PfSlam(Node):

    def __init__(self):
        super().__init__('pf_slam')
        self.log = self.get_logger()

        self.declare_parameter('alpha_v', 0.02)
        self.declare_parameter('alpha_w', 0.02) 
        self.declare_parameter('meas_sigma_d_a', 0.017340)
        self.declare_parameter('meas_sigma_d_b', 0.001990)
        self.declare_parameter('meas_sigma_t_a', 0.000104)
        self.declare_parameter('meas_sigma_t_b', 0.000088)

        #### Loading Map #####
        map_file = os.path.expanduser('~/ros2_ws/src/prob_rob_labs_ros_2/prob_rob_labs/config/landmarks.yaml')
        if not os.path.exists(map_file):
            self.log.error(f"Landmark file does not exist at path:\n{map_file}")
            self.landmarks = {}
            return
        self.log.info(f"Loading landmark map from: {map_file}")
        with open(map_file, 'r') as f:
            self.landmarks = yaml.safe_load(f)
        self.log.info(f"Loaded landmarks: {self.landmarks}")

        #### Subscriptions and Publishers #####
        self.camera_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        self.measurement_pubs = {}
        for color, info in self.landmarks["landmarks"].items():
            meas_topic = f"/measurement_{color}"
            self.measurement_pubs[color] = self.create_publisher(Point2DArrayStamped, meas_topic, 10)

            vision_topic = f"/vision_{color}/corners"
            self.create_subscription(
                Point2DArrayStamped,
                vision_topic,
                lambda msg, color=color, info=info: self.process_corner_message(msg, color, info),
                10
            )

            self.log.info(f"Subscribed to {vision_topic} and publishing to {meas_topic}")

        #### Parameter declaring #####
        self.p_matrix = None
        self.img_h = None
        self.img_w = None 
        self.system_time = None

    def estimate_distance_bearing(self, corners_msg, landmark_height):
        points = corners_msg.points
        if not points or len(points) < min_points:
            return None, None

        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])

        x_min, x_20, x_80, x_max = float(np.min(x)), float(np.percentile(x, 20)), float(np.percentile(x, 80)), float(np.max(x))
        y_min, y_20, y_80, y_max = float(np.min(y)), float(np.percentile(y, 20)), float(np.percentile(y, 80)), float(np.max(y))

        x_margin = min([x_min, self.img_w - x_max])
        y_margin = min([y_min, self.img_h - y_max])

        if x_margin < self.img_w * min_x_margin_percent:
            return None, None

        if (y_max > self.img_h * y_max_percent) or (y_margin < self.img_h * min_y_margin_percent):
            return None, None

        x_center_perceived = (x_80 + x_20) / 2.0
        height_perceived = y_max - y_min

        cx = self.p_matrix[0][2]
        fx = self.p_matrix[0][0]
        fy = self.p_matrix[1][1]

        bearing = np.arctan((cx - x_center_perceived) / fx)
        distance = landmark_height * fy / (height_perceived * np.cos(bearing))
        return distance, bearing

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
            ### insert pf motion update calling function
            pass

        self.system_time = t_meas

        ### insert pf measurement update step (call a function to do that?)

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
    
    
    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    pf_slam = PfSlam()
    pf_slam.spin()
    pf_slam.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
