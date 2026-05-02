#!/usr/bin/env python3
"""
Puzzlebot Robot Controller
===========================
Subscribes to:
    /cmd_vel  (geometry_msgs/Twist) — velocity command

Publishes:
    /VelocitySetR  (std_msgs/Float32) — right wheel angular velocity [rad/s]
    /VelocitySetL  (std_msgs/Float32) — left  wheel angular velocity [rad/s]

Converts a Twist command into individual wheel velocities using the
differential-drive kinematic model. Works in simulation and on the real robot.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


WHEEL_RADIUS = 0.05   # metres — must match model.sdf
WHEEL_BASE   = 0.18   # metres — must match model.sdf robot_width


class PuzzlebotController(Node):

    def __init__(self):
        super().__init__('puzzlebot_controller')

        self.declare_parameter('wheel_radius', WHEEL_RADIUS)
        self.declare_parameter('wheel_base',   WHEEL_BASE)

        self.r = self.get_parameter('wheel_radius').value
        self.L = self.get_parameter('wheel_base').value

        self.pub_right = self.create_publisher(Float32, '/VelocitySetR', 10)
        self.pub_left  = self.create_publisher(Float32, '/VelocitySetL', 10)

        self.create_subscription(Twist, '/cmd_vel', self.cb_cmd_vel, 10)

        self.get_logger().info(
            f'Puzzlebot controller ready — '
            f'wheel_radius={self.r} m, wheel_base={self.L} m'
        )

    def cb_cmd_vel(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z

        # Differential-drive inverse kinematics → wheel angular velocities
        vel_r = (2.0 * v + w * self.L) / (2.0 * self.r)
        vel_l = (2.0 * v - w * self.L) / (2.0 * self.r)

        msg_r = Float32()
        msg_l = Float32()
        msg_r.data = float(vel_r)
        msg_l.data = float(vel_l)

        self.pub_right.publish(msg_r)
        self.pub_left.publish(msg_l)


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()