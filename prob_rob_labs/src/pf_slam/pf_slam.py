import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from prob_rob_msgs.msg import Point2DArrayStamped, Point2D
from nav_msgs.msg import Odometry

import os
import yaml
import numpy as np
import math

min_points = 4
min_x_margin_percent = 0.02
min_y_margin_percent = 0.03
y_max_percent = 0.625

CAM_OFFSET = np.array([0.076, 0, 0.093])

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

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

        self.create_subscription(Odometry, '/ekf_odom', self.odom_callback, 10)

        #### Parameter declaring #####
        self.p_matrix = None
        self.img_h = None
        self.img_w = None 
        self.system_time = None
        self.last_odom_time = None

        self.last_vel = 0.0
        self.last_omega = 0.0

        self.alpha_v = self.get_parameter('alpha_v').value
        self.alpha_w = self.get_parameter('alpha_w').value
        self.meas_sigma_d_a = self.get_parameter('meas_sigma_d_a').value
        self.meas_sigma_d_b = self.get_parameter('meas_sigma_d_b').value
        self.meas_sigma_t_a = self.get_parameter('meas_sigma_t_a').value
        self.meas_sigma_t_b = self.get_parameter('meas_sigma_t_b').value

        self.num_particles = 100
        self.particles = []
        self.initialize_particles()

    def initialize_particles(self):
        self.particles = []
        initial_theta = 0.0
        initial_x = -1.5
        initial_y = 0.0

        for _ in range (self.num_particles):
            theta = initial_theta + np.random.normal(0, 0.05)
            x = initial_x + np.random.normal(0, 0.1)
            y = initial_y + np.random.normal(0, 0.1)

            particle = {
                "theta": theta,
                "x": x, 
                "y": y,
                "weight": 1.0 / self.num_particles,
                "landmarks": {}
            }
            self.particles.append(particle)

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

        ### pf measurement update function
        self.pf_update(color, dist, bearing)


        ### sending as topic for debugging purpose
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
    
    def odom_callback(self, odom_msg: Odometry):
        t_odom = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec*1e-9
        vel = odom_msg.twist.twist.linear.x
        omega = odom_msg.twist.twist.angular.z

        if self.last_odom_time is None:
            self.last_odom_time = t_odom
            self.last_vel = vel
            self.last_omega = omega
            return

        dt = t_odom - self.last_odom_time
        self.last_odom_time = t_odom

        if dt < 0:
            self.log.warn("Late odometry sample ignored")
            return
        if dt == 0:
            self.last_vel = odom_msg.twist.twist.linear.x
            self.last_omega = odom_msg.twist.twist.angular.z
            return

        self.last_vel = vel
        self.last_omega = omega

        self.pf_predict(dt)
        self.log.info("Valid odom update: dt={dt:.3f}, v={self.last_vel:.3f}, omega={self.last_omega:.3f}")

    def pf_predict(self, dt: float):
        ### to propagate all particles forward
        
        if self.last_vel is None or self.last_omega is None:
            return ## haven't received odometry yet
        
        v = self.last_vel
        omega = self.last_omega

        sigma_v = math.sqrt(self.alpha_v * (v ** 2 + 1e-6))
        sigma_w = math.sqrt(self.alpha_w * (omega ** 2 + 1e-6))

        EPS_ANG = 1e-6

        for p in self.particles:
            theta = p["theta"]
            x = p["x"]
            y = p["y"]

            v_sample = v + np.random.normal(0.0, sigma_v)
            w_sample = omega + np.random.normal(0.0, sigma_w)

            if abs(w_sample) < EPS_ANG:
                dtheta = 0.0
                dx = v_sample * dt * math.cos(theta)
                dy = v_sample * dt * math.cos(theta)
            else:
                dtheta = w_sample * dt
                dx = (v_sample / w_sample) * (math.sin(theta + w_sample * dt) - math.sin(theta))
                dy = (v_sample / w_sample) * (-math.cos(theta + w_sample * dt) + math.cos(theta))

            theta_new = wrap_angle(theta + dtheta)

            p["theta"] = theta_new
            p["x"] = x + dx
            p["y"] = y + dy

    def pf_update(self, color: str, dist: float, bearing: float):
        '''
            measurement update for 1 landmark at a time;
            per particle landmark update

        '''
        if self.p_matrix is None:
            return
        
        if not self.particles:
            return
        
        sigma_d2 = self.meas_sigma_d_a + self.meas_sigma_d_b * dist
        sigma_t2 = self.meas_sigma_t_a + self.meas_sigma_t_b * dist
        R = np.diag([sigma_d2, sigma_t2])

        z = np.array([[dist], [bearing]])

        for p in self.particles:
            theta = p["theta"]
            x_r = p["x"]
            y_r = p["y"]

            dx_cam = CAM_OFFSET[0]
            dy_cam = CAM_OFFSET[1]

            x_cam = x_r + math.cos(theta) * dx_cam - math.sin(theta) * dy_cam
            y_cam = y_r + math.sin(theta) * dx_cam + math.cos(theta) * dy_cam

            if color not in p["landmarks"]:
                global_bearing = theta + bearing
                lm_x = x_cam + dist * math.cos(global_bearing)
                lm_y = y_cam + dist * math.sin(global_bearing)

                mu = np.array([[lm_x], [lm_y]])

                # taking jacobian J = dm/dz
                J = np.zeros((2, 2))
                J[0, 0] = math.cos(global_bearing)
                J[0, 1] = -dist * math.sin(global_bearing)
                J[1, 0] = math.sin(global_bearing)
                J[1, 1] =  dist * math.cos(global_bearing)

                sigma = J @ R @ J.T

                p["landmarks"][color] = {
                    "mu": mu,
                    "Sigma": sigma
                    }
                continue

            lm = p["landmarks"][color]
            mu = lm["mu"]        
            sigma = lm["Sigma"]  

            lm_x = float(mu[0, 0])
            lm_y = float(mu[1, 0])

            dx_l = lm_x - x_cam
            dy_l = lm_y - y_cam

            q = dx_l**2 + dy_l**2
            if q < 1e-9:
                continue
            pred_dist = math.sqrt(q)
            pred_bearing = math.atan2(dy_l, dx_l) - theta
            pred_bearing = wrap_angle(pred_bearing)

            z_pred = np.array([[pred_dist],
                            [pred_bearing]])
            
            # Innovation Step
            y = z - z_pred
            y[1, 0] = wrap_angle(y[1, 0])

            # Jacobian wrt landmark position
            r = pred_dist
            H = np.zeros((2, 2))
            H[0, 0] = dx_l / r
            H[0, 1] = dy_l / r
            H[1, 0] = -dy_l / q
            H[1, 1] =  dx_l / q

            S = H @ sigma @ H.T + R

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue

            K = sigma @ H.T @ S_inv

            # EKF landmark update
            mu_new = mu + K @ y
            sigma_new = (np.eye(2) - K @ H) @ sigma

            lm["mu"] = mu_new
            lm["Sigma"] = sigma_new

            # measurement likelihood for this particle

            exponent = -0.5 * float(y.T @ S_inv @ y)
            likelihood = math.exp(exponent)

            p["weight"] *= likelihood

        # normalizing particle weights
        total_w = sum(p["weight"] for p in self.particles)
        if total_w <= 0.0:
            uniform_w = 1.0/len(self.particles)
            for p in self.particles:
                p["weight"] = uniform_w
        else:
            for p in self.particles:
                p["weight"] /= total_w

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
