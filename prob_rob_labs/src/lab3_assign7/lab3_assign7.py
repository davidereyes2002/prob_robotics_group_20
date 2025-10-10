import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Float64
import time

heartbeat_period = 0.1

class ManualDoorCycler(Node):

    def __init__(self):
        super().__init__('manual_door_cycler')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        self.pub_door_open = self.create_publisher(Empty, '/door_open', 10)
        self.pub_door_torque = self.create_publisher(Float64, '/hinged_glass_door/torque', 10)

        self.declare_parameter('close_torque', -5.0)

        self.wait_after_trigger = 5.0
        self.close_duration = 3.0
        self.cooldown = 1.0

        self.state = 'IDLE'
        self.t0 = None
        self.trial = 0
        self.max_trials = 32

    def heartbeat(self):
        if self.trial >= self.max_trials and self.state != 'DONE':
            self.state = 'DONE'
            self.log.info('STATE DONE')
            self.pub_door_torque.publish(Float64(data=0.0))
            return

        if self.state == 'IDLE':
            self.pub_door_open.publish(Empty())
            self.log.info(f'Trial {self.trial + 1}: Published /door_open')
            self.t0 = time.time()
            self.state = 'WAIT'
            self.log.info('STATE WAIT')

        elif self.state == 'WAIT':
            if time.time() - self.t0 >= self.wait_after_trigger:
                torque = self.get_parameter('close_torque').get_parameter_value().double_value
                self.pub_door_torque.publish(Float64(data=torque))
                self.log.info(f'Applying close torque: {torque}')
                self.t0 = time.time()
                self.state = 'CLOSE'
                self.log.info('STATE CLOSE')

        elif self.state == 'CLOSE':
            if time.time() - self.t0 >= self.close_duration:
                self.pub_door_torque.publish(Float64(data=0.0))
                self.log.info('Torque reset to 0. Door should be closed.')
                self.t0 = time.time()
                self.state = 'COOLDOWN'
                self.log.info('STATE COOLDOWN')

        elif self.state == 'COOLDOWN':
            if time.time() - self.t0 >= self.cooldown:
                self.trial += 1
                self.state = 'IDLE'
                self.log.info('STATE IDLE (Next cycle starting...)')

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = ManualDoorCycler()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
