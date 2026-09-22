"""Synthesizes nav_msgs/Odometry from Isaac Sim's separate pose/twist topics.

ego-planner's EGOReplanFSM subscribes to a single nav_msgs/Odometry topic
(pose + linear/angular velocity). Isaac Sim instead publishes PoseStamped
and TwistStamped on separate topics, so this node time-synchronizes them
and republishes a combined Odometry message.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

import message_filters
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry


class OdomBridgeNode(Node):

    def __init__(self):
        super().__init__('isaac_odom_bridge')

        self.declare_parameter('pose_topic', 'state/pose')
        # Switch to 'state/twist' here to A/B test against gps_twist and
        # confirm which one is the body-frame vs. world-frame velocity.
        self.declare_parameter('twist_topic', 'state/twist_inertial')
        self.declare_parameter('output_topic', 'visual_slam/odom')
        self.declare_parameter('world_frame_id', 'world')
        self.declare_parameter('child_frame_id', '')
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.02)

        pose_topic = self.get_parameter('pose_topic').value
        twist_topic = self.get_parameter('twist_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.world_frame_id = self.get_parameter('world_frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        queue_size = self.get_parameter('sync_queue_size').value
        slop = self.get_parameter('sync_slop').value

        # Isaac Sim's ROS2 bridge publishes these as BEST_EFFORT/VOLATILE;
        # a default RELIABLE subscription silently receives nothing.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=queue_size,
        )

        self.pose_sub = message_filters.Subscriber(
            self, PoseStamped, pose_topic, qos_profile=sensor_qos)
        self.twist_sub = message_filters.Subscriber(
            self, TwistStamped, twist_topic, qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.twist_sub], queue_size, slop)
        self.sync.registerCallback(self.sync_callback)

        # ego-planner subscribes with the default (RELIABLE) QoS.
        self.odom_pub = self.create_publisher(Odometry, output_topic, 10)

        self.get_logger().info(
            f"isaac_odom_bridge: {pose_topic} + {twist_topic} -> {output_topic} "
            f"(slop={slop}s)")

    def sync_callback(self, pose_msg: PoseStamped, twist_msg: TwistStamped):
        odom = Odometry()
        # Isaac Sim's vehicle-state publishers and its Render/Replicator
        # camera publishers stamp messages from two unrelated clocks (one
        # looked like sim-elapsed seconds, the other like wall-clock unix
        # epoch) -- differing by ~1.8e9 seconds. grid_map's ApproximateTime
        # sync of depth+odom can never match across that gap, so
        # occupancy_inflate stays permanently empty regardless of what's
        # in view. Re-stamping here with our own node's clock (shared with
        # depth_restamp_node's own now()) puts both back in one domain,
        # off only by real processing latency (sub-ms to a few ms).
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.world_frame_id or pose_msg.header.frame_id
        odom.child_frame_id = self.child_frame_id or twist_msg.header.frame_id

        odom.pose.pose = pose_msg.pose
        odom.twist.twist = twist_msg.twist

        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = OdomBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
