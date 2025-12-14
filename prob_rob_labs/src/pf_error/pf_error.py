import rclpy
from rclpy.node import Node


heartbeat_period = 0.1

class PfError(Node):

    def __init__(self):
        super().__init__('pf_error')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

    def heartbeat(self):
        self.log.info('heartbeat')

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    pf_error = PfError()
    pf_error.spin()
    pf_error.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
