#!/usr/bin/env python3
"""PuzzleBot teleop node for keyboard-controlled robot velocity commands."""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select
import threading


class PuzzlebotTeleop(Node):
    """Teleop node for PuzzleBot with smooth acceleration."""

    def __init__(self):
        """
        Initialize the teleop node with velocity publishers and parameters.

        Sets up ROS 2 velocity publishers, declares configurable speed and
        acceleration parameters, and initializes the keyboard input handler.
        """
        super().__init__('puzzlebot_teleop')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # 1. ROS 2 Parameters for configurable speeds and acceleration
        self.declare_parameter('max_lin', 0.8)
        self.declare_parameter('max_ang', 0.8)
        self.declare_parameter('accel_lin', 0.2)
        self.declare_parameter('accel_ang', 0.05)

        self.max_lin = self.get_parameter('max_lin').value
        self.max_ang = self.get_parameter('max_ang').value
        self.accel_lin = self.get_parameter('accel_lin').value
        self.accel_ang = self.get_parameter('accel_ang').value

        # Target velocities (what the user wants based on key press)
        self.target_lin = 0.0
        self.target_ang = 0.0

        # Current velocities (what the robot is actually doing)
        self.current_lin = 0.0
        self.current_ang = 0.0

        # 50Hz update rate for smooth velocity publishing
        self.timer = self.create_timer(0.02, self.publish_velocity)

        self.get_logger().info(
            "\nTeleop Active:\n"
            "---------------------------\n"
            "  W / S : Linear Move\n"
            "  A / D : Angular Turn\n"
            "  Space : Emergency Stop\n"
            "  'q' or CTRL+C to quit.\n"
            "---------------------------"
        )

    def process_key(self, key):
        """Map standard keystrokes to target velocities."""
        key = key.lower()
        if key == 'w':
            self.target_lin = self.max_lin
            self.target_ang = 0.0
        elif key == 's':
            self.target_lin = -self.max_lin
            self.target_ang = 0.0
        elif key == 'a':
            self.target_lin = 0.0
            self.target_ang = self.max_ang
        elif key == 'd':
            self.target_lin = 0.0
            self.target_ang = -self.max_ang
        elif key == ' ':
            # Hard stop
            self.target_lin = 0.0
            self.target_ang = 0.0
            self.current_lin = 0.0
            self.current_ang = 0.0

    def publish_velocity(self):
        """
        Interpolate current velocity toward target (Kinematic Smoothing).

        Smoothly ramps linear and angular velocities towards their target
        values using acceleration limits to create kinematic smoothing.
        """
        # Smoothly ramp linear velocity
        if self.target_lin > self.current_lin:
            self.current_lin = min(
                self.target_lin,
                self.current_lin +
                self.accel_lin)
        elif self.target_lin < self.current_lin:
            self.current_lin = max(
                self.target_lin,
                self.current_lin -
                self.accel_lin)

        # Smoothly ramp angular velocity
        if self.target_ang > self.current_ang:
            self.current_ang = min(
                self.target_ang,
                self.current_ang +
                self.accel_ang)
        elif self.target_ang < self.current_ang:
            self.current_ang = max(
                self.target_ang,
                self.current_ang -
                self.accel_ang)

        # Publish
        msg = Twist()
        msg.linear.x = self.current_lin
        msg.angular.z = self.current_ang
        self.publisher_.publish(msg)

    def key_loop(self):
        """Dedicated loop to handle blocking terminal input safely."""
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok():
                # 0.1s timeout acts as an automatic key-release detector
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    key = sys.stdin.read(1)
                    if key == '\x03' or key.lower() == 'q':  # CTRL+C or 'q'
                        break
                    self.process_key(key)
                else:
                    # Timeout reached (no keys pressed) -> coast to a stop
                    self.target_lin = 0.0
                    self.target_ang = 0.0
        finally:
            # Always guarantee terminal settings are restored upon exiting loop
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    def stop_robot(self):
        """Stop the robot by zeroing out velocities before shutdown."""
        msg = Twist()
        self.publisher_.publish(msg)


def main(args=None):
    """
    Run the teleop node.

    Initializes the PuzzlebotTeleop node and spins the ROS 2 event loop
    in a background thread while handling keyboard input in the main thread.
    """
    rclpy.init(args=args)
    node = PuzzlebotTeleop()

    # 2. Spin ROS 2 callbacks in a background thread to unblock terminal
    # reading
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # Run the blocking keyboard reader in the main thread
        node.key_loop()
    except KeyboardInterrupt:
        pass
    finally:
        # Graceful shutdown pipeline
        print("\rExiting teleop node...         ")
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    