#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select

class PuzzlebotTeleop(Node):
    def __init__(self):
        super().__init__('puzzlebot_teleop')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.max_lin = 0.2
        self.max_ang = 0.8 
        
        # Save terminal settings to restore on exit
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        # 50Hz update rate
        self.timer = self.create_timer(0.02, self.teleop_loop)
        
        self.get_logger().info("Teleop Active: W/S (Linear), A/D (Angular). Press 'q' or CTRL+C to quit.")

    def get_key(self):
        """Non-blocking key read."""
        try:
            tty.setraw(sys.stdin.fileno())
            # Use select with a tiny timeout for that "game" feel
            rlist, _, _ = select.select([sys.stdin], [], [], 0.04)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            # Restore settings immediately so CTRL+C works
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        return key

    def teleop_loop(self):
        key = self.get_key()
        msg = Twist()

        # FIXED INVERSION: 
        # If 'w' and 's' were turning, we assign them to linear.x
        # If 'a' and 'd' were moving forward, we assign them to angular.z
        if key == 'w':
            msg.linear.x = self.max_lin
            msg.angular.z = 0.0
        elif key == 's':
            msg.linear.x = -self.max_lin
            msg.angular.z = 0.0
        elif key == 'a':
            msg.linear.x = 0.0
            msg.angular.z = self.max_ang
        elif key == 'd':
            msg.linear.x = 0.0
            msg.angular.z = -self.max_ang
        elif key == 'q' or key == '\x03': # 'q' or CTRL+C
            self.get_logger().info("Exiting...")
            rclpy.shutdown()
        else:
            # Release key = Stop
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotTeleop()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt):
        pass
    finally:
        # Final safety stop
        node.publisher_.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.old_settings)
        node.destroy_node()

if __name__ == '__main__':
    main()