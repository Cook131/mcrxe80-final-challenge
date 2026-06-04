#!/usr/bin/env python3
"""
puzzlebotGoToGoal.py — PID Go-To-Goal controller
=================================================
Sits in the pipeline:
  A* ──/goal──► GoToGoal ──/cmd_raw──► bug_IBA ──/cmd_vel──► Controller

Publishes to /cmd_raw (NOT /cmd_vel) so the bug_IBA reflex layer
can override commands when obstacles are detected.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Bool


class GoToGoalNode(Node):

    def __init__(self):
        super().__init__('go_to_goal_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('kp_v',                  1.8)
        self.declare_parameter('ki_v',                  0.02)
        self.declare_parameter('kd_v',                  0.1)
        self.declare_parameter('kp_w',                  1.0)
        self.declare_parameter('ki_w',                  0.0)
        self.declare_parameter('kd_w',                  0.0)
        self.declare_parameter('max_linear_velocity',   0.22)
        self.declare_parameter('max_angular_velocity',  0.5)
        self.declare_parameter('angle_threshold',       0.15)
        self.declare_parameter('goal_reached_dist',     0.10)  # 10 cm — realistic for hardware

        self.kp_v   = self.get_parameter('kp_v').value
        self.ki_v   = self.get_parameter('ki_v').value
        self.kd_v   = self.get_parameter('kd_v').value
        self.kp_w   = self.get_parameter('kp_w').value
        self.ki_w   = self.get_parameter('ki_w').value
        self.kd_w   = self.get_parameter('kd_w').value
        self.max_v  = self.get_parameter('max_linear_velocity').value
        self.max_w  = self.get_parameter('max_angular_velocity').value
        self.angle_threshold  = self.get_parameter('angle_threshold').value
        self.goal_reached_dist = self.get_parameter('goal_reached_dist').value

        # ── State ─────────────────────────────────────────────────────────
        self.x   = 0.0
        self.y   = 0.0
        self.th  = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.active   = False
        # Suspendido por bug_IBA durante evasión BUG2
        self._nav_paused = False

        self.error_dist_prev  = 0.0
        self.error_angle_prev = 0.0
        self.integral_dist    = 0.0
        self.integral_angle   = 0.0
        self.last_time        = self.get_clock().now()

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom',      self._cb_odom,      10)
        self.create_subscription(Pose2D,   '/goal',      self._cb_goal,      10)
        self.create_subscription(Bool,     '/nav_pause', self._cb_nav_pause, 10)

        # ── Publisher → /cmd_raw so bug_IBA can intercept ─────────────────
        self._pub_cmd = self.create_publisher(Twist, '/cmd_raw', 10)

        self.get_logger().info(
            f'GoToGoal started — goal_reached_dist={self.goal_reached_dist} m, '
            f'publishing to /cmd_raw'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_nav_pause(self, msg: Bool):
        """Pausa/reanuda el controlador según señal de bug_IBA."""
        was_paused = self._nav_paused
        self._nav_paused = msg.data
        if msg.data and not was_paused:
            # Publicar stop inmediato al pausar para que /cmd_raw quede en cero
            self._pub_cmd.publish(Twist())
            self.get_logger().warn('[GoToGoal] NAV PAUSADA — BUG2 evadiendo obstáculo')
        elif not msg.data and was_paused:
            self.get_logger().info('[GoToGoal] NAV REANUDADA — retomando control')

    def _cb_odom(self, msg: Odometry):
        self.x  = msg.pose.pose.position.x
        self.y  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.th = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        if self.active and not self._nav_paused:
            self._control_loop()

    def _cb_goal(self, msg: Pose2D):
        self.target_x = msg.x
        self.target_y = msg.y
        self.active   = True

        # Reset integrators on new goal
        self.integral_dist    = 0.0
        self.integral_angle   = 0.0
        self.error_dist_prev  = 0.0
        self.error_angle_prev = 0.0

        self.get_logger().info(
            f'New goal: ({self.target_x:.2f}, {self.target_y:.2f})'
        )

    # ── Control loop ──────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
        self.last_time = now

        dist        = math.hypot(self.target_x - self.x, self.target_y - self.y)
        angle_goal  = math.atan2(self.target_y - self.y, self.target_x - self.x)
        error_angle = math.atan2(
            math.sin(angle_goal - self.th),
            math.cos(angle_goal - self.th)
        )

        cmd = Twist()

        if dist < self.goal_reached_dist:
            # Stop and deactivate — A* will send next waypoint if any
            self.active = False
            self.get_logger().info('✅ Goal reached')
            self._pub_cmd.publish(cmd)
            return

        # ── Angular PID ───────────────────────────────────────────────────
        self.integral_angle   += error_angle * dt
        derivative_angle       = (error_angle - self.error_angle_prev) / dt
        w_out = (self.kp_w * error_angle
                 + self.ki_w * self.integral_angle
                 + self.kd_w * derivative_angle)
        cmd.angular.z = max(min(w_out, self.max_w), -self.max_w)

        # ── Linear PID — only drive forward when roughly aligned ──────────
        if abs(error_angle) > self.angle_threshold:
            cmd.linear.x = 0.0
        else:
            self.integral_dist  += dist * dt
            derivative_dist      = (dist - self.error_dist_prev) / dt
            v_out = (self.kp_v * dist
                     + self.ki_v * self.integral_dist
                     + self.kd_v * derivative_dist)
            align_factor   = math.cos(error_angle)
            cmd.linear.x   = min(v_out * align_factor, self.max_v)

        self.error_dist_prev  = dist
        self.error_angle_prev = error_angle

        self._pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GoToGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()