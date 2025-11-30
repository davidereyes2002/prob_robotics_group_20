import rclpy
from rclpy.node import Node
import yaml
import os
import numpy as np

from sensor_msgs.msg import CameraInfo
from prob_rob_msgs.msg import Point2DArrayStamped, Point2D

Min_pts = 4
Min_x_margin_percentage = 0.02
Min_y_margin_percentage = 0.03
y_max_percentage = 0.625

class EkfLocalization(Node):

    def __init__(self):
        super().__init__('ekf_localization')
        self.log = self.get_logger()

        self.declare_parameter('map_file', '')
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

        dist, theta = self.estimate_distance_bearing(corners_msg, height)

        if dist is None or theta is None:
            return

        msg = Point2DArrayStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"

        pt = Point2D()
        pt.x = float(dist)
        pt.y = float(theta)
        msg.points = [pt]

        self.measurement_pubs[color].publish(msg)
        self.log.info(
            f"[{color}] Published: distance={dist:.3f} | bearing={theta:.3f}"
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

        theta = np.arctan((cx - x_center_perceived) / fx)
        distance = landmark_height * fy / (height_perceived * np.cos(theta))

        return distance, theta

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
