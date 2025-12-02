import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from prob_rob_msgs.msg import Point2DArrayStamped, Point2D
import numpy as np
import message_filters

Min_pts = 4
Min_x_margin_percentage = 0.02
Min_y_margin_percentage = 0.03
y_max_percentage = 0.625

class Lab5Assign2(Node):

    def __init__(self):
        super().__init__('lab5_assign2')
        
        self.declare_parameter('landmark_target', 'cyan')
        self.landmark_target = self.get_parameter('landmark_target').get_parameter_value().string_value
        self.get_logger().info(f"Landmark target color set to: {self.landmark_target} \n")

        self.declare_parameter('landmark_height', 0.5)
        self.landmark_height = self.get_parameter('landmark_height').value
        
        self.p_matrix = None
        self.img_w = None
        self.img_h = None

        self.camera_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)

        self.target_vision_topic = f"/vision_{self.landmark_target}/corners"
        self.corners_sub = self.create_subscription(Point2DArrayStamped, self.target_vision_topic, self.corners_callback, 10)
        
        self.target_pose_topic = f'/estimated_pose/{self.landmark_target}'
        self.pub_range_bearing = self.create_publisher(Point2DArrayStamped, self.target_pose_topic, 10)

    def camera_info_callback(self, msg: CameraInfo):
        if self.p_matrix is not None:
            return

        self.img_w = msg.width
        self.img_h = msg.height
        self.p_matrix = np.array(msg.p).reshape(3, 4)[:, :3]

        self.destroy_subscription(self.camera_sub)
        self.camera_sub = None

    def corners_callback(self, msg: Point2DArrayStamped):
        if self.p_matrix is None:
            self.get_logger().warn("CameraInfo not yet received.")
            return

        distance, theta = self.estimate_distance_bearing(msg)

        if distance is None or theta is None:
            self.get_logger().info(f"[{self.landmark_target}] Invalid measurement. Skipping.")
            return

        msg_out = Point2DArrayStamped()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.header.frame_id = 'camera_link'

        pt = Point2D()
        pt.x = float(distance)
        pt.y = float(theta)
        msg_out.points = [pt]

        self.pub_range_bearing.publish(msg_out)
        self.get_logger().info(f"[{self.landmark_target}] d={distance:.3f} m | θ={theta:.3f} rad")
    
    def estimate_distance_bearing(self, corners_msg):
        points = corners_msg.points

        if not points or len(points) < Min_pts:
            self.get_logger().warn(f"No {self.landmark_target} landmark detected.")
            return None, None

        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        x_min, x_20, x_80, x_max = float(np.min(x)), float(np.percentile(x, 20)), float(np.percentile(x, 80)), float(np.max(x))
        y_min, y_20, y_80, y_max = float(np.min(y)), float(np.percentile(y, 20)), float(np.percentile(y, 80)), float(np.max(y))

        x_distance_margin = min(x_min, self.img_w - x_max)
        y_distance_margin = min(y_min, self.img_h - y_max)
        
        if x_distance_margin < self.img_w * Min_x_margin_percentage:
            self.get_logger().warn(f"Left or right side of {self.landmark_target} landmark is out of sight.")
            return None, None
        elif (y_max > self.img_h * y_max_percentage) or (y_distance_margin < self.img_h * Min_y_margin_percentage):
            self.get_logger().warn(f"Up or bottom side of {self.landmark_target} landmark is out of sight.")
            return None, None
        
        x_center_perceived = (x_80 + x_20) / 2.0
        height_perceived = y_max - y_min
        cx = self.p_matrix[0][2]
        fx = self.p_matrix[0][0]
        fy = self.p_matrix[1][1]
        theta = np.arctan((cx - x_center_perceived)/fx)
        distance = self.landmark_height * fy /(height_perceived * np.cos(theta))
        self.get_logger().info(
            f"height_px={height_perceived:.2f}, x_center={x_center_perceived:.2f}, "
            f"theta={theta:.3f}, d={distance:.3f}"
        )

        return distance, theta
    
    def spin(self):
        rclpy.spin(self)

def main():
    rclpy.init()
    lab5_assign2 = Lab5Assign2()
    lab5_assign2.spin()
    lab5_assign2.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
