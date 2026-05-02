#!/usr/bin/env python3
"""
Puzzlebot Odometry Node — Dead Reckoning
=========================================
Subscribes to:
    /VelocityEncR  (std_msgs/Float32)  — right wheel angular velocity [rad/s]
    /VelocityEncL  (std_msgs/Float32)  — left  wheel angular velocity [rad/s]

Publishes:
    /odom          (nav_msgs/Odometry) — robot pose and velocity estimate

Works both in Gazebo simulation and on the real Puzzlebot hardware,
since both use the same topic names and message types.

Dead-reckoning model (differential drive):
    v  = wheel_radius * (wr + wl) / 2        linear  velocity  [m/s]
    w  = wheel_radius * (wr - wl) / wheel_base  angular velocity [rad/s]
    x  += v * cos(theta) * dt
    y  += v * sin(theta) * dt
    theta += w * dt
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster

import math


# ── Robot physical constants ───────────────────────────────────────────────────
# Match these to the values in model.sdf → DiffDynamicPlugin
WHEEL_RADIUS = 0.05   # metres  (wheel_radius in plugin)
WHEEL_BASE   = 0.18   # metres  (robot_width  in plugin)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert roll/pitch/yaw (radians) to a geometry_msgs/Quaternion."""
    cy = math.cos(yaw   * 0.5)
    sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class OdometryNode(Node):

    def __init__(self):
        super().__init__('odometry_node')

        # ── Parameters (easy to override via CLI or launch) ────────────────
        self.declare_parameter('wheel_radius', WHEEL_RADIUS)
        self.declare_parameter('wheel_base',   WHEEL_BASE)
        self.declare_parameter('publish_rate',  100.0)   # Hz

        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_base   = self.get_parameter('wheel_base').value
        publish_rate      = self.get_parameter('publish_rate').value

        # ── State ──────────────────────────────────────────────────────────
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        self.wr = 0.0   # right wheel angular velocity [rad/s]
        self.wl = 0.0   # left  wheel angular velocity [rad/s]

        self.last_time = self.get_clock().now()

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            Float32,
            '/VelocityEncR',
            self.cb_enc_right,
            10
        )
        self.create_subscription(
            Float32,
            '/VelocityEncL',
            self.cb_enc_left,
            10
        )

        # ── Publisher ──────────────────────────────────────────────────────
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # ── TF broadcaster (odom → base_link) ─────────────────────────────
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Timer ──────────────────────────────────────────────────────────
        self.create_timer(1.0 / publish_rate, self.update)

        self.get_logger().info(
            f'Odometry node started — '
            f'wheel_radius={self.wheel_radius} m, '
            f'wheel_base={self.wheel_base} m, '
            f'rate={publish_rate} Hz'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def cb_enc_right(self, msg: Float32):
        self.wr = msg.data

    def cb_enc_left(self, msg: Float32):
        self.wl = msg.data

    # ── Main update loop ───────────────────────────────────────────────────

    def update(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0:
            return

        # Convert angular wheel velocities → robot body velocities
        v = self.wheel_radius * (self.wr + self.wl) / 2.0          # linear  [m/s]
        w = self.wheel_radius * (self.wr - self.wl) / self.wheel_base  # angular [rad/s]

        # Integrate pose
        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta += w * dt

        # Normalise angle to [-π, π]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Build and publish Odometry message
        odom = Odometry()
        odom.header.stamp    = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        # Pose
        odom.pose.pose.position.x  = self.x
        odom.pose.pose.position.y  = self.y
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = euler_to_quaternion(0.0, 0.0, self.theta)

        # Twist (velocities in the robot frame)
        odom.twist.twist.linear.x  = v
        odom.twist.twist.linear.y  = 0.0
        odom.twist.twist.angular.z = w

        # Covariance — diagonal, modest uncertainty for dead reckoning.
        # Indices follow the 6x6 row-major layout: [x, y, z, roll, pitch, yaw]
        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[35] = 0.05   # yaw
        odom.twist.covariance[0]  = 0.01  # vx
        odom.twist.covariance[35] = 0.05  # wz

        self.odom_pub.publish(odom)

        # ── Broadcast odom → base_link TF ─────────────────────────────────
        t = TransformStamped()
        t.header.stamp    = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation      = euler_to_quaternion(0.0, 0.0, self.theta)
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()