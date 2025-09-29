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
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)
        self.start_time = None

        self.declare_parameter('forward_speed', 0.5)
        self.declare_parameter('open_torque', 2.0)
        self.declare_parameter('close_torque', -2.0)

        self.declare_parameter('open_time', 10.0)
        self.declare_parameter('move_time', 10.0)
        self.declare_parameter('stop_time', 5.0)
        self.declare_parameter('close_time', 10.0)

        # Precompute time thresholds
        self.open_time   = self.get_parameter('open_time').value
        self.move_time   = self.get_parameter('move_time').value
        self.stop_time   = self.get_parameter('stop_time').value
        self.close_time  = self.get_parameter('close_time').value

        self.state_times = [
            self.open_time,
            self.move_time,
            self.stop_time,
            self.close_time,
        ]
        self.state_cumulative_times = [sum(self.state_times[:i+1]) for i in range(len(self.state_times))]

    def heartbeat(self):
        if self.start_time is None:
            self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
            return
        
        t = self.get_clock().now().seconds_nanoseconds()[0] - self.start_time

        forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        open_torque   = self.get_parameter('open_torque').get_parameter_value().double_value
        close_torque  = self.get_parameter('close_torque').get_parameter_value().double_value
        
        torque = Float64()
        twist = Twist()

        if t < self.state_cumulative_times[0]:
            torque.data = open_torque
            self.publisher_door.publish(torque)
            self.log.info('Opening door')

        elif t < self.state_cumulative_times[1]:
            twist.linear.x = forward_speed
            self.publisher_cmd_vel.publish(twist)
            self.log.info(f'Moving through door at speed {forward_speed}')

        elif t < self.state_cumulative_times[2]:
            twist.linear.x = 0.0
            self.publisher_cmd_vel.publish(twist)
            self.log.info('Stopping robot')

        elif t < self.state_cumulative_times[3]:
            torque.data = close_torque
            self.publisher_door.publish(torque)
            self.log.info('Closing door')

        else:
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
