"""Republishes a depth Image with its header.stamp overwritten to this
node's own clock, so it lands in the same clock domain as
odom_bridge_node's output.

Isaac Sim's Render/Replicator camera publishers and its vehicle-state
publishers stamp messages from two unrelated clocks (one looked like
sim-elapsed seconds, the other like wall-clock unix epoch), off by
roughly 1.8e9 seconds. grid_map's message_filters::ApproximateTime sync
of depth+odom can never match a pair across that gap, so
occupancy_inflate stays permanently empty no matter what's actually in
view. Re-stamping both sides with each bridge node's own now() fixes
that, at the cost of the sub-ms to few-ms delay between Isaac publishing
the frame and this node relaying it.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class DepthRestampNode(Node):

    def __init__(self):
        super().__init__('depth_restamp_node')

        self.declare_parameter('input_topic', 'camera/depth')
        self.declare_parameter('output_topic', 'camera/depth_restamped')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # /drone<id>/camera/depth is published RELIABLE/VOLATILE by
        # Isaac's Render/Replicator writer -- matches the subscription
        # default here.
        depth_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # grid_map's depth_sub_ also defaults to RELIABLE.
        self.pub = self.create_publisher(Image, output_topic, depth_qos)
        self.sub = self.create_subscription(
            Image, input_topic, self.cb, depth_qos)

        self.get_logger().info(f'depth_restamp_node: {input_topic} -> {output_topic}')

    def cb(self, msg: Image):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthRestampNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
