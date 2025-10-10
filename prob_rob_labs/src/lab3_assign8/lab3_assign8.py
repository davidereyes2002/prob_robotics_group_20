import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Empty
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import time

heartbeat_period = 0.1
class BayesDoorController(Node):

    def __init__(self):
        super().__init__('lab3_a8')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        self.publisher_randdoor = self.create_publisher(Empty, '/door_open', 10)
        self.publisher_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mean_sub = self.create_subscription(Float64, '/feature_mean', self.subscriber_callback, 10)

        self.open_door_threshold = 240.0
        self.feature_mean_value = None

        self.declare_parameter('close_torque', -5.0)

        self.p_open_open_push = 1.0
        self.p_close_open_push = 0.6875
        self.p_open_close_push = 1 - self.p_close_open_push
        self.p_close_open_push = 0.0

        #declaring constants
        self.p_z_open_given_open = 0.610
        self.p_z_open_given_closed = 0.0
        self.p_z_closed_given_open = 1 - self.p_z_open_given_open
        self.p_z_closed_given_closed = 1.0

        self.belief = 0.5
        self.target_confidence = 0.99

        #state machine
        self.state = 'IDLE'
        self.t0 = None

        self.push_duration = 5.0
        self.push_duration_2 = 5.0
        self.measure_duration = 0.8
        self.max_steps = 30
        self.steps = 0

    def subscriber_callback(self, msg: Float64):
        self.feature_mean_value = msg.data

    
    def bayes_predict(self, b):
        b_pred = self.p_open_open_push * b + self.p_close_open_push * (1 - b)
        return b_pred
    
    def bayes_update(self, b_pred, z_open):
        if z_open:
            num = self.p_z_open_given_open * b_pred
            den = num + self.p_z_open_given_closed * (1 - b_pred)
        else:
            num = self.p_z_closed_given_open * b_pred
            den = num + self.p_z_closed_given_closed * (1 - b_pred)
        return num / den if den > 0 else b_pred

    def heartbeat(self):
        twist = Twist()

        if self.state == 'IDLE' and self.belief < self.target_confidence and self.steps < self.max_steps:
            self.log.info('STATE: IDLE')
            self.publisher_randdoor.publish(Empty())
            self.log.info('PUSHED')
            self.t0 = time.time()
            self.belief = self.bayes_predict(self.belief)
            self.log.info(f'Predicted belief after push: {self.belief:.3f}')
            self.state = 'WAIT'
            self.log.info('STATE: WAIT')

        elif self.state == 'WAIT':
            if time.time() - self.t0 >= self.push_duration:
                self.t0 = time.time()
                self.state = 'MEASURE'
                self.log.info('STATE: MEASURE')

        elif self.state == 'WAIT_2':
            if time.time() - self.t0 >= self.push_duration_2:
                self.t0 = time.time()
                self.state = 'DONE'
                self.log.info('STATE: DONE')

        elif self.state == 'MEASURE':
            if self.feature_mean_value is not None:
                z_open = (self.feature_mean_value <= self.open_door_threshold)
                self.belief = self.bayes_update(self.belief, z_open)
                self.steps += 1
                self.log.info(f'Measurement: door {"OPEN" if z_open else "CLOSED"} | Belief(open)={self.belief:.3f}')
                self.state = 'IDLE'

                # self.log.info(f'Step={self.steps} measured open or not: {z_open} | belief(open)={self.belief:.3f}')
                # self.state = 'IDLE'
                if self.belief >= self.target_confidence:
                    self.log.info(f'Door confirmed open! Belief={self.belief:.3f} ≥ {self.target_confidence}')
                    twist.linear.x = 1.5
                    self.t0 = time.time()                    
                    self.publisher_cmd_vel.publish(twist)
                    self.state = 'WAIT_2'
                    self.log.info('STATE: WAIT_2')

        elif self.state == 'DONE':
            twist.linear.x = 0.0
            self.publisher_cmd_vel.publish(twist)
            return

        elif self.steps >= self.max_steps:
            self.log.warn('Reached max steps without high confidence.')
            self.state = 'DONE'

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = BayesDoorController()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
