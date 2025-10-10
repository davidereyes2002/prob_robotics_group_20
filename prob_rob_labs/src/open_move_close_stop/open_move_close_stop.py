import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

heartbeat_period = 0.1

class OpenMoveCloseStop(Node):

    def __init__(self):
        super().__init__('open_move_close_stop')
        self.publisher_door = self.create_publisher(Float64, '/hinged_glass_door/torque', 10)
        self.publisher_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mean_sub = self.create_subscription(Float64, '/feature_mean', self.subscriber_callback, 1)
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)
        self.start_time = None

        self.state = 0
        self.feature_mean_value = None
        self.open_door_threshold = 235.0

        self.declare_parameter('forward_speed', 0.5)
        self.declare_parameter('open_torque', 1.5)
        self.declare_parameter('close_torque', -1.5)

        self.declare_parameter('move_time', 8.0)
        self.declare_parameter('stop_time', 2.0)
        self.declare_parameter('close_time', 10.0)

        self.move_time   = self.get_parameter('move_time').value
        self.stop_time   = self.get_parameter('stop_time').value
        self.close_time  = self.get_parameter('close_time').value

        self.state_times = [
            self.move_time,
            self.stop_time,
            self.close_time,
        ]
        self.state_cumulative_times = [sum(self.state_times[:i+1]) for i in range(len(self.state_times))]
        
        self.p_open = 0.5
        self.z_open_x_open = 0.610
        self.z_closed_x_open = 1 - self.z_open_x_open
        self.z_closed_x_closed = 1
        self.z_open_x_closed = 1 - self.z_closed_x_closed
        self.open_belief_threshold = 0.9999

    def subscriber_callback(self, msg: Float64):
        self.feature_mean_value = msg.data

    def heartbeat(self):
        if self.start_time is not None:
            t = self.get_clock().now().seconds_nanoseconds()[0] - self.start_time

        forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        open_torque   = self.get_parameter('open_torque').get_parameter_value().double_value
        close_torque  = self.get_parameter('close_torque').get_parameter_value().double_value
        
        torque = Float64()
        twist = Twist()

        if self.state == 0:
            torque.data = open_torque
            self.publisher_door.publish(torque)
            self.log.info(f'Opening door. Current state: {self.state}')
            self.log.info(f'This is the feature_mean value: {self.feature_mean_value}')

            if self.feature_mean_value is not None:
                if self.feature_mean_value < self.open_door_threshold:
                    self.p_open = self.z_open_x_open * self.p_open * (1/(self.z_open_x_open * self.p_open + self.z_open_x_closed * (1-self.p_open)))
                else:
                    self.p_open = self.z_closed_x_open * self.p_open * (1/(self.z_closed_x_open * self.p_open + self.z_closed_x_closed * (1-self.p_open)))
            
                if self.p_open > self.open_belief_threshold:
                    torque.data = 0.0
                    self.publisher_door.publish(torque)
                    self.state = 1
                    self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
                    self.log.info(f'Door is detected to be opened. Transitioning to state {self.state}')
            else:
                self.log.warn("Feature mean not received yet, waiting...")

        elif self.state == 1:
            twist.linear.x = forward_speed
            self.publisher_cmd_vel.publish(twist)
            self.log.info(f'Moving through door at speed {forward_speed}. Current state: {self.state}')

            if t > self.state_cumulative_times[0]:
                self.state = 2
                self.log.info(f'Robot has fully moved. Transitioning to state {self.state}')

        elif self.state == 2:
            twist.linear.x = 0.0
            self.publisher_cmd_vel.publish(twist)
            self.log.info(f'Stopping robot. Current state: {self.state}')

            if t > self.state_cumulative_times[1]:
                self.state = 3
                self.log.info(f'Robot has fully stopped. Transitioning to state {self.state}')

        elif self.state == 3:
            torque.data = close_torque
            self.publisher_door.publish(torque)
            self.log.info(f'Closing door. Current state: {self.state}')

            if t > self.state_cumulative_times[2]:
                self.state = 4
                self.log.info(f'Door is fully closed. Transitioning to state {self.state}')

        elif self.state == 4:
            torque.data = 0.0
            self.publisher_door.publish(torque)
            self.log.info('Finished!')
            

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    open_move_close_stop = OpenMoveCloseStop()
    open_move_close_stop.spin()
    open_move_close_stop.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()