"""Forwards geometry_msgs/Twist body-frame velocity commands to ArduPilot
SITL over MAVLink using SET_POSITION_TARGET_LOCAL_NED (velocity + yaw-rate
only, MAV_FRAME_BODY_OFFSET_NED).

Twist is in ROS's FLU body convention (X-forward, Y-left, Z-up, positive
yaw-rate = CCW from above). MAV_FRAME_BODY_OFFSET_NED is FRD (X-forward,
Y-right, Z-down, positive yaw-rate = CW from above). Confirmed directly
from Isaac Sim's own published tf: drone_base_link -> drone_base_link_frd
is a 180 deg rotation about X (quaternion x=1,y=0,z=0,w=0), which negates
Y and Z and, applied to angular velocity, negates yaw-rate too -- so all
three of linear.y, linear.z and angular.z need the sign flip, not just z.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pymavlink import mavutil


class CmdVelBridge(Node):

    def __init__(self):
        super().__init__('cmd_vel_mavlink_bridge')

        self.declare_parameter('connection_string', 'udp:127.0.0.1:14550')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')

        connection_string = self.get_parameter('connection_string').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.get_logger().info(f'Connecting to {connection_string} ...')
        self.master = mavutil.mavlink_connection(connection_string)
        self.master.wait_heartbeat()
        self.get_logger().info(
            f'Heartbeat received (system {self.master.target_system} '
            f'component {self.master.target_component})')

        self.sub = self.create_subscription(Twist, cmd_vel_topic, self.cb, 10)

    def cb(self, msg: Twist):
        self.get_logger().debug(
            f'linear: x={msg.linear.x:.2f} y={msg.linear.y:.2f} z={msg.linear.z:.2f} '
            f'| angular: z={msg.angular.z:.2f}')
        # type_mask: enable only vx, vy, vz, yaw_rate (ignore position/accel/yaw)
        type_mask = 0b0000_0111_1100_0111
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            0, 0, 0,           # x, y, z (ignored)
            msg.linear.x, -msg.linear.y, -msg.linear.z,  # FLU->FRD: y,z flip
            0, 0, 0,           # accel (ignored)
            0, -msg.angular.z,  # yaw, yaw_rate (FLU CCW -> FRD CW)
        )


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
