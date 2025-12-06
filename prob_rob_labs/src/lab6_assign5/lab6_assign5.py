import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from prob_rob_msgs.msg import Point2DArrayStamped


class MapOdomTfNode(Node):

    def __init__(self):
        super().__init__('map_odom_tf_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.ekf_sub = self.create_subscription(
            Odometry,
            '/ekf_pose',
            self.ekf_callback,
            10
        )

        self.latest_map_odom_tf: TransformStamped | None = None

        self.last_meas_time = None
        for color in ["red", "green", "yellow", "magenta", "cyan"]:
            topic = f"/measurement_{color}"
            self.create_subscription(
                Point2DArrayStamped,
                topic,
                self.measurement_callback,
                10
            )

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.get_logger().info('map_odom_tf_node initialized')

    def measurement_callback(self, msg: Point2DArrayStamped):
        self.last_meas_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    
    def pose_to_matrix(self, pose):
        tx = pose.position.x
        ty = pose.position.y
        tz = pose.position.z
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
        R[0, 1] = 2 * (qx * qy - qz * qw)
        R[0, 2] = 2 * (qx * qz + qy * qw)

        R[1, 0] = 2 * (qx * qy + qz * qw)
        R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
        R[1, 2] = 2 * (qy * qz - qx * qw)

        R[2, 0] = 2 * (qx * qz - qy * qw)
        R[2, 1] = 2 * (qy * qz + qx * qw)
        R[2, 2] = 1 - 2 * (qx * qx + qy * qy)

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [tx, ty, tz]
        return T

    def tf_to_matrix(self, tf_msg: TransformStamped):
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        class P:
            pass

        pose = P()
        pose.position = P()
        pose.orientation = P()

        pose.position.x = t.x
        pose.position.y = t.y
        pose.position.z = t.z
        pose.orientation.x = q.x
        pose.orientation.y = q.y
        pose.orientation.z = q.z
        pose.orientation.w = q.w

        return self.pose_to_matrix(pose)

    def rotation_matrix_to_quaternion(self, R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return qx, qy, qz, qw

    def matrix_to_tf(self, T, parent, child, stamp):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent
        tf_msg.child_frame_id = child

        tf_msg.transform.translation.x = float(T[0, 3])
        tf_msg.transform.translation.y = float(T[1, 3])
        tf_msg.transform.translation.z = float(T[2, 3])

        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(T[0:3, 0:3])
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        return tf_msg

    def ekf_callback(self, msg: Odometry):
        if self.last_meas_time is None:
            return
        
        t_ekf = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = abs(t_ekf - self.last_meas_time)
        if dt > 0.05:
            return
        
        try:
            tf_odom_base = self.tf_buffer.lookup_transform(
                'odom', 'base_footprint', rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f"lookup odom->base_footprint failed: {e}"
            )
            return

        T_map_base = self.pose_to_matrix(msg.pose.pose)

        T_odom_base = self.tf_to_matrix(tf_odom_base)

        T_base_odom = np.linalg.inv(T_odom_base)
        T_map_odom = T_map_base @ T_base_odom

        self.latest_map_odom_tf = self.matrix_to_tf(
            T_map_odom,
            parent='map',
            child='odom',
            stamp=msg.header.stamp
        )

    def timer_callback(self):
        if self.latest_map_odom_tf is None:
            return

        self.latest_map_odom_tf.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.latest_map_odom_tf)


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomTfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()