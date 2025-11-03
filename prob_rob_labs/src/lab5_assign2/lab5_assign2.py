import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from prob_rob_msgs.msg import Point2DArrayStamped, Point2D
import numpy as np
import message_filters

class Lab5Assign2(Node):

    def __init__(self):
        super().__init__('lab5_assign2')
        self.camera_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera_info')
        
        self.landmark_target = "cyan"
        self.target_vision_topic = f"/vision_{self.landmark_target}/corners"
        self.landmark_height = 0.5
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
        raw_p_matrix = cam_msg.p
        self.p_matrix = self.extract_p_matrix(raw_p_matrix)
        self.get_logger().info("CameraInfo message extracted and synced")
        
        points = corners_msg.points
        if not points:
            self.get_logger().warn("No corners detected for cyan landmark.")
            return
        else:
            self.height, self.x_center = self.calculate_height_and_x_center(points)
            self.get_logger().info(f"Vision Corners received and processed")
    
        self.distance, self.theta = self.estimate_distance_bearing()
        self.get_logger().info(f"Estimated distance: {self.distance}, estimated theta: {self.theta}")

        msg_out = Point2DArrayStamped()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.header.frame_id = 'camera_link'
        pt = Point2D()
        pt.x = float(self.distance)
        pt.y = float(self.theta)
        msg_out.points = [pt]

        self.pub_range_bearing.publish(msg_out)
        self.get_logger().info(f"Published d={self.distance:.3f} m, θ={self.theta:.3f} rad")

    def calculate_height_and_x_center(self, points):
        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        height = np.max(y) - np.min(y)
        x_center = (np.max(x) + np.min(x)) / 2.0
        return height, x_center
    
    def extract_p_matrix(self, raw_p_matrix):
        p_matrix = np.array(raw_p_matrix).reshape(3, 4)[:, :3]
        return p_matrix
    
    def estimate_distance_bearing(self):
        theta = np.arctan((self.p_matrix[0][2]-self.x_center)/self.p_matrix[0][0])
        distance = self.landmark_height * self.p_matrix[1][1]/(self.height * np.cos(theta))
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
