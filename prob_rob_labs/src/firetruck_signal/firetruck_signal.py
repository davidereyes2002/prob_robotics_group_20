import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64

heartbeat_period = 0.1

class FiretruckSignal(Node):

    def __init__(self):
        super().__init__('firetruck_signal')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        self.publisher_door = self.create_publisher(Float64, '/hinged_glass_door/torque', 10)

        self.mean_sub = self.create_subscription(Float64, '/feature_mean', self.subscriber_callback, 1)
        self.open_door_threshold = 235.0
        

        self.feature_mean_value = None

        self.declare_parameter('open_torque', 5.0)
        self.declare_parameter('close_torque', -5.0)

        self.state = 'MEASURE_CLOSE'
        self.t0 = None

        self.gt_close = np.array([])
        self.gt_open = np.array([])

        self.meas_close = np.array([])
        self.meas_open = np.array([])

        self.track = 0

    def subscriber_callback(self, msg: Float64):
        self.feature_mean_value = msg.data

    def heartbeat(self):
        # self.log.info('heartbeat')
        open_torque   = self.get_parameter('open_torque').get_parameter_value().double_value

        if self.t0 == None:
            self.t0 = self.get_clock().now().nanoseconds * 1e-9
            self.log.info('STATE: MEASURE_CLOSE')

        torque = Float64()
        torque.data = open_torque
        
        if self.state == 'MEASURE_CLOSE':
            
            if self.get_clock().now().nanoseconds * 1e-9 - self.t0 <= 10.0:
                # self.log.info(f'Feature mean value: {self.feature_mean_value}')
                if self.feature_mean_value is not None:
                    self.gt_close = np.append(self.gt_close, self.feature_mean_value)
            else:
                self.state = 'PUSH'
                self.log.info('STATE: PUSH')
                self.t0 = self.get_clock().now().nanoseconds * 1e-9
        
        elif self.state == 'PUSH':
                self.publisher_door.publish(torque)
                self.t0 = self.get_clock().now().nanoseconds * 1e-9
                self.state = 'WAIT'
                self.log.info('STATE: WAIT')

        elif self.state == 'WAIT':
            if self.get_clock().now().nanoseconds * 1e-9 - self.t0 >= 5.0:
                self.state = 'MEASURE_OPEN'
                self.log.info('STATE: MEASURE_OPEN')
                self.t0 = self.get_clock().now().nanoseconds * 1e-9 

        elif self.state == 'MEASURE_OPEN' and self.track == 0:
            if self.get_clock().now().nanoseconds * 1e-9 - self.t0 <= 10.0:
                # self.log.info(f'Feature mean value: {self.feature_mean_value}')
                self.gt_open = np.append(self.gt_open, self.feature_mean_value)
            else:
                z_open_when_close = self.gt_close[self.gt_close < self.open_door_threshold]
                z_close_when_close = self.gt_close[self.gt_close >= self.open_door_threshold]
                z_open_when_open = self.gt_open[self.gt_open < self.open_door_threshold]
                z_close_when_open = self.gt_open[self.gt_open >= self.open_door_threshold]

                FP = len(z_open_when_close)
                TN = len(z_close_when_close)
                TP = len(z_open_when_open)
                FN = len(z_close_when_open)

                x_closed_total = FP + TN
                x_open_total   = TP + FN

                p_zopen_given_xclosed    = (FP / x_closed_total) if x_closed_total > 0 else float('nan')
                p_zclosed_given_xclosed  = (TN / x_closed_total) if x_closed_total > 0 else float('nan')
                p_zopen_given_xopen      = (TP / x_open_total)   if x_open_total   > 0 else float('nan')
                p_zclosed_given_xopen    = (FN / x_open_total)   if x_open_total   > 0 else float('nan')

                self.log.info(
                    f'Finished.\n'
                    f'Counts: TP={TP} FP={FP} TN={TN} FN={FN}\n'
                    f'P(z=open|x=closed)={p_zopen_given_xclosed:.3f}  '
                    f'P(z=closed|x=open)={p_zclosed_given_xopen:.3f}  '
                    f'P(z=open|x=open)={p_zopen_given_xopen:.3f}  '
                    f'P(z=closed|x=closed)={p_zclosed_given_xclosed:.3f}'
                )
                self.track = 1


    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    firetruck_signal = FiretruckSignal()
    firetruck_signal.spin()
    firetruck_signal.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()