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

        self.camera_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera_info')
        
        self.declare_parameter('landmark_target', 'cyan')
        self.landmark_target = self.get_parameter('landmark_target').get_parameter_value().string_value
        self.get_logger().info(f"Landmark target color set to: {self.landmark_target} \n")

        self.target_vision_topic = f"/vision_{self.landmark_target}/corners"
        self.declare_parameter('landmark_height', 0.5)
        self.landmark_height = self.get_parameter('landmark_height').value
        self.corners_sub = message_filters.Subscriber(self, Point2DArrayStamped, self.target_vision_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.camera_info_sub, self.corners_sub], queue_size=10, slop=0.1)
        
        self.height = None
        self.x_center = None
        self.p_matrix = None
        self.theta = None
        self.distance = None
        
        self.ts.registerCallback(self.synced_callback)
        self.target_pose_topic = f'/estimated_pose/{self.landmark_target}'
        self.pub_range_bearing = self.create_publisher(Point2DArrayStamped, self.target_pose_topic, 10)


    def synced_callback(self, cam_msg: CameraInfo, corners_msg: Point2DArrayStamped):

        self.distance, self.theta = self.estimate_distance_bearing(cam_msg, corners_msg)

        if self.distance or self.theta:
            msg_out = Point2DArrayStamped()
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.header.frame_id = 'camera_link'
            pt = Point2D()
            pt.x = float(self.distance)
            pt.y = float(self.theta)
            msg_out.points = [pt]

            self.pub_range_bearing.publish(msg_out)
            self.get_logger().info(f"Estimation Published d={self.distance:.3f} m, θ={self.theta:.3f} rad. \n")
        else:
            self.get_logger().info("Estimation Invalid, Skip Publishing. \n")
    
    def estimate_distance_bearing(self, cam_msg: CameraInfo, corners_msg: Point2DArrayStamped):
        points = corners_msg.points
        img_w = cam_msg.width
        img_h = cam_msg.height
        p_matrix = np.array(cam_msg.p).reshape(3, 4)[:, :3]

        if not points or len(points) < Min_pts:
            self.get_logger().warn(f"No {self.landmark_target} landmark detected.")
            theta = None
            distance = None
            return distance, theta

        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        x_min, x_20, x_80, x_max = float(np.min(x)), float(np.percentile(x, 20)), float(np.percentile(x, 80)), float(np.max(x))
        y_min, y_20, y_80, y_max = float(np.min(y)), float(np.percentile(y, 20)), float(np.percentile(y, 80)), float(np.max(y))
        x_distance_margin = min([x_min, img_w-x_max])
        y_distance_margin = min([y_min, img_h-y_max])
        
        if x_distance_margin < img_w * Min_x_margin_percentage:
            self.get_logger().warn(f"Left or right side of {self.landmark_target} landmark is out of sight.")
            theta = None
            distance = None
        elif (y_max > img_h * y_max_percentage) or (y_distance_margin < img_h * Min_y_margin_percentage):
            self.get_logger().warn(f"Up or bottom side of {self.landmark_target} landmark is out of sight.")
            theta = None
            distance = None
            
        else:
            x_center_perceived = (x_80 + x_20) / 2.0
            height_perceived = y_max - y_min
            cx = p_matrix[0][2]
            fx = p_matrix[0][0]
            fy = p_matrix[1][1]
            theta = np.arctan((cx - x_center_perceived)/fx)
            distance = self.landmark_height * fy /(height_perceived * np.cos(theta))
            self.get_logger().info(f"{self.landmark_target} Landmark Is Valid Within Sight")
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
