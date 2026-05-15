#!/usr/bin/env python3
"""
puzzlebotController.py — Closed-loop wheel velocity controller.

==============================================================
Architecture
------------
  /cmd_vel (Twist)
      |
      v inverse kinematics
  [target_wr, target_wl]  <- desired wheel angular velocities [rad/s]
      |
      v  PID (one per wheel, runs at 50 Hz)
  [VelocitySetR, VelocitySetL]  -> firmware PWM driver
      ^
  [VelocityEncR, VelocityEncL]  <- actual wheel velocities from encoders

Why a PID?
----------
The original node was pure open-loop: it sent a set-point and trusted
the firmware to hit it exactly.  On the real Puzzlebot, motor friction,
load, and battery voltage make the actual speed differ from the set-point.
That error accumulates in odometry, blurring the SLAM map.

The PID closes the loop:
  error  = target_velocity - measured_velocity
  output = Kp*error + Ki*int(error dt) + Kd*d(error)/dt

Tuning guide (start here, adjust on the bench)
-----------------------------------------------
  Kp = 1.0   -> raise until wheels reach set-point quickly without overshoot
  Ki = 0.5   -> raise slowly to eliminate steady-state error at low speeds
  Kd = 0.05  -> raise slightly only if you see oscillation

Override via --ros-args:
  ros2 run iolair controller --ros-args -p Kp:=1.2 -p Ki:=0.6 -p Kd:=0.02

Subscribes to:
    /cmd_vel       (geometry_msgs/Twist)  - velocity command
    /VelocityEncR  (std_msgs/Float32)     - measured right wheel [rad/s]
    /VelocityEncL  (std_msgs/Float32)     - measured left  wheel [rad/s]

Publishes to:
    /VelocitySetR  (std_msgs/Float32)     - right wheel set-point [rad/s]
    /VelocitySetL  (std_msgs/Float32)     - left  wheel set-point [rad/s]
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# ── Simple PID ──────────────────────────────────────────────────────────

class PID:
    """
    Discrete PID controller with anti-windup and output clamping.

    Parameters
    ----------
    Kp, Ki, Kd : gains
    dt         : time step [s] — must match the control-loop rate
    out_min    : minimum output value (clamps integral windup too)
    out_max    : maximum output value

    """

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 dt: float, out_min: float, out_max: float):
        """Initialize PID controller with gains and output limits."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max

        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Reset integral and previous error tracking."""
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float) -> float:
        """Compute PID output for given error value."""
        # Proportional
        p = self.Kp * error

        # Integral with anti-windup (clamp before accumulating)
        self._integral += error * self.dt
        i_raw = self.Ki * self._integral
        # Clamp integral term to output limits to prevent windup
        i_clamped = max(self.out_min, min(self.out_max, i_raw))
        self._integral = i_clamped / self.Ki if self.Ki != 0.0 else 0.0

        # Derivative (on measurement, not error, to avoid derivative kick)
        d = self.Kd * (error - self._prev_error) / self.dt
        self._prev_error = error

        output = p + i_clamped + d
        return max(self.out_min, min(self.out_max, output))

    def update_gains(self, Kp: float, Ki: float, Kd: float):
        """Update PID gains and reset controller state."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.reset()


# ── Controller Node ─────────────────────────────────────────────────────

class PuzzlebotController(Node):
    """Closed-loop wheel velocity controller for Puzzlebot robot."""

    # Physical constants (match model.sdf / odometry node)
    WHEEL_RADIUS = 0.05   # metres
    WHEEL_BASE = 0.19   # metres

    # Maximum wheel angular velocity the motors can deliver [rad/s]
    # RPM_max ≈ 200 rpm → 200/60 * 2π ≈ 21 rad/s (conservative cap)
    MAX_WHEEL_VEL = 21.0

    def __init__(self):
        """Initialize controller with PID gains and ROS interfaces."""
        super().__init__('puzzlebot_main_controller')

        # ── PID gains (tunable at runtime via parameters) ─────────────────
        self.declare_parameter('Kp', 1.0)
        self.declare_parameter('Ki', 0.5)
        self.declare_parameter('Kd', 0.05)
        self.declare_parameter('control_rate', 50.0)   # Hz

        Kp = self.get_parameter('Kp').value
        Ki = self.get_parameter('Ki').value
        Kd = self.get_parameter('Kd').value
        rate = self.get_parameter('control_rate').value
        dt = 1.0 / rate

        self._pid_r = PID(Kp, Ki, Kd, dt,
                          -self.MAX_WHEEL_VEL, self.MAX_WHEEL_VEL)
        self._pid_l = PID(Kp, Ki, Kd, dt,
                          -self.MAX_WHEEL_VEL, self.MAX_WHEEL_VEL)

        # ── Desired wheel velocities (set by /cmd_vel callback) ───────────
        self._target_r = 0.0
        self._target_l = 0.0

        # ── Measured wheel velocities (set by encoder callbacks) ──────────
        self._meas_r = 0.0
        self._meas_l = 0.0

        # ── Current control output (integrates PID output into set-point) ─
        # We start at zero and let the PID drive toward the target.
        self._set_r = 0.0
        self._set_l = 0.0

        # ── QoS: match the firmware's BEST_EFFORT encoder publishers ──────
        enc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Twist, '/cmd_vel', self._cb_cmd_vel, 10)
        self.create_subscription(
            Float32,
            '/VelocityEncR',
            self._cb_enc_r,
            enc_qos)
        self.create_subscription(
            Float32,
            '/VelocityEncL',
            self._cb_enc_l,
            enc_qos)

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)
        self._pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)

        # ── Control loop timer ────────────────────────────────────────────
        self.create_timer(dt, self._control_loop)

        self.get_logger().info(
            f'PuzzlebotController ready — '
            f'Kp={Kp}, Ki={Ki}, Kd={Kd}, rate={rate} Hz'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_cmd_vel(self, msg: Twist):
        """Convert Twist to desired wheel angular velocities [rad/s]."""
        v = msg.linear.x
        w = msg.angular.z
        r = self.WHEEL_RADIUS
        L = self.WHEEL_BASE

        # Differential-drive inverse kinematics
        self._target_r = (2.0 * v + w * L) / (2.0 * r)
        self._target_l = (2.0 * v - w * L) / (2.0 * r)

        # Clamp to motor limits
        self._target_r = max(-self.MAX_WHEEL_VEL,
                             min(self.MAX_WHEEL_VEL, self._target_r))
        self._target_l = max(-self.MAX_WHEEL_VEL,
                             min(self.MAX_WHEEL_VEL, self._target_l))

        # Reset integral when target changes sign (avoids wind-up on reversal)
        if self._target_r * self._pid_r._prev_error < -0.5:
            self._pid_r.reset()
        if self._target_l * self._pid_l._prev_error < -0.5:
            self._pid_l.reset()

    def _cb_enc_r(self, msg: Float32):
        """Process right encoder measurement."""
        self._meas_r = msg.data

    def _cb_enc_l(self, msg: Float32):
        """Process left encoder measurement."""
        self._meas_l = msg.data

    # ── Control loop ────────────────────────────────────────────────────────

    def _control_loop(self):
        """Run control loop to compute PID outputs and publish set-points."""
        # When target is zero, stop immediately (don't let PID fight a stop)
        if self._target_r == 0.0 and self._target_l == 0.0:
            self._pid_r.reset()
            self._pid_l.reset()
            self._publish(0.0, 0.0)
            return

        error_r = self._target_r - self._meas_r
        error_l = self._target_l - self._meas_l

        # PID output is a correction; add it to the current set-point
        # (velocity-form PID — smoother than position-form for motor control)
        self._set_r = float(self._pid_r.update(error_r))
        self._set_l = float(self._pid_l.update(error_l))

        self._publish(self._set_r, self._set_l)

    # ── Publisher helper ────────────────────────────────────────────────────

    def _publish(self, vel_r: float, vel_l: float):
        """Publish wheel velocity set-points to firmware."""
        msg_r = Float32()
        msg_l = Float32()
        msg_r.data = vel_r
        msg_l.data = vel_l
        self._pub_r.publish(msg_r)
        self._pub_l.publish(msg_l)


# ── Entry point ─────────────────────────────────────────────────────────

def main(args=None):
    """
    Start the Puzzlebot controller node.

    Initializes the PuzzlebotController and runs the ROS 2 event loop
    until interrupted by user input. Performs safe shutdown with
    velocity set-points zeroed.
    """
    rclpy.init(args=args)
    node = PuzzlebotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Safety stop on shutdown
        node._publish(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
