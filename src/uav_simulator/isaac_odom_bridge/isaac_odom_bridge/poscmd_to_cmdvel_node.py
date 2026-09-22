"""Converts ego-planner's quadrotor_msgs/PositionCommand (traj_server's
output) into geometry_msgs/Twist so cmd_vel_mavlink_bridge can forward it
to ArduPilot.

cmd_vel_mavlink_bridge sends velocity setpoints in MAV_FRAME_BODY_OFFSET_NED
(body frame), but PositionCommand.velocity is in the world frame. This node
rotates it into body frame using the current yaw from odom.

Twist has no field for an absolute yaw setpoint, only angular.z (yaw
rate), so yaw is tracked with feedforward + a P term on yaw error:
    angular.z = kp * wrap(yaw_cmd - yaw_now) + yaw_dot_cmd
rather than passing yaw_dot straight through, which would let yaw error
drift and accumulate over a long trajectory.

No timer is needed here: traj_server publishes /position_cmd continuously
at 100Hz (via its own wall timer) regardless of whether a trajectory is
active, so relaying on every message already gives ArduPilot the
continuous setpoint stream its GUIDED velocity control needs.
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PosCmdToCmdVelNode(Node):

    def __init__(self):
        super().__init__('poscmd_to_cmdvel_node')

        self.declare_parameter('position_cmd_topic', '/position_cmd')
        self.declare_parameter('odom_topic', 'visual_slam/odom')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('yaw_kp', 1.5)

        position_cmd_topic = self.get_parameter('position_cmd_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.yaw_kp = self.get_parameter('yaw_kp').value

        self.yaw_now = None

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_cb, 10)
        self.poscmd_sub = self.create_subscription(
            PositionCommand, position_cmd_topic, self.poscmd_cb, 50)
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.get_logger().info(
            f'poscmd_to_cmdvel_node: {position_cmd_topic} + {odom_topic} '
            f'-> {cmd_vel_topic}')

    def odom_cb(self, msg: Odometry):
        self.yaw_now = yaw_from_quaternion(msg.pose.pose.orientation)

    def poscmd_cb(self, msg: PositionCommand):
        if self.yaw_now is None:
            return  # no odom yet -- don't command anything blind

        vx_world = msg.velocity.x
        vy_world = msg.velocity.y

        cos_yaw = math.cos(self.yaw_now)
        sin_yaw = math.sin(self.yaw_now)
        vx_body = vx_world * cos_yaw + vy_world * sin_yaw
        vy_body = -vx_world * sin_yaw + vy_world * cos_yaw

        yaw_error = wrap_to_pi(msg.yaw - self.yaw_now)
        yaw_rate = self.yaw_kp * yaw_error + msg.yaw_dot

        twist = Twist()
        twist.linear.x = vx_body
        twist.linear.y = vy_body
        twist.linear.z = msg.velocity.z
        twist.angular.z = yaw_rate
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = PosCmdToCmdVelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
