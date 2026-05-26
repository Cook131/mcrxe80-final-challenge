#!/usr/bin/env python3
"""
puzzlebotOdometry_EKF.py — Full EKF Odometry with automatic MCL / ICP switching
=================================================================================

Measurement source selection
-----------------------------
Two external pose sources are supported:

  SOURCE A — /mcl/pose  (PoseWithCovarianceStamped)
      Published by puzzlebotMCL.py when localising on a known map.

  SOURCE B — /icp/pose  (PoseWithCovarianceStamped)
      Published by slam_node when building a map with ICP scan matching.

The node monitors BOTH topics.  Whichever one has published a message
within the last `source_timeout` seconds is considered **active**.
If both are active simultaneously, MCL takes priority (it is the more
accurate source once a map exists).

Source switching is automatic and logged clearly so you can see in the
terminal which source the EKF is currently fusing.

State machine
-------------
  PREDICT_ONLY  → no external source alive
  MCL_ACTIVE    → /mcl/pose  alive  (localisation mode)
  ICP_ACTIVE    → /icp/pose  alive  (mapping mode)
  MCL_PRIORITY  → both alive        (MCL wins)

Fallback behaviour
------------------
  When the active source goes silent (robot turned off, node crash, etc.)
  the EKF reverts to pure dead-reckoning automatically.  The covariance
  will start growing again — the node logs a WARNING so you notice.

EKF equations
-------------
PREDICT (50 Hz, encoder-driven):
    x̂⁻  = f(x̂, u)              differential drive Euler integration
    P⁻   = F @ P @ Fᵀ + Q       linearised covariance propagation

UPDATE (on each incoming measurement):
    y    = z − x̂⁻               innovation  (H = I → direct pose)
    S    = P⁻ + R                innovation covariance
    K    = P⁻ @ S⁻¹              Kalman gain
    x̂    = x̂⁻ + K @ y           state correction
    P    = (I−K)P(I−K)ᵀ+KRKᵀ   Joseph-form covariance update
    P    = (P+Pᵀ)/2              symmetry enforcement

Parameters (--ros-args -p name:=value)
--------------------------------------
  wheel_radius      0.05   [m]
  wheel_base        0.19   [m]
  rate              50.0   [Hz]
  q_xy              0.005  process noise – translation  [m²/s]
  q_theta           0.01   process noise – rotation     [rad²/s]
  source_timeout    0.5    seconds of silence before a source is declared dead
  r_pos_default     0.1    fallback R diagonal for xy   [m²]
  r_yaw_default     0.05   fallback R diagonal for yaw  [rad²]
  max_innov_pos     1.0    innovation gate – position   [m]
  max_innov_yaw     1.5    innovation gate – yaw        [rad]

Publishes
---------
  /odom                  (nav_msgs/Odometry)
  /ekf/active_source     (std_msgs/String)   current source name for monitoring
  TF: odom → base_link

Subscribes
----------
  /VelocityEncL    (std_msgs/Float32)
  /VelocityEncR    (std_msgs/Float32)
  /mcl/pose        (geometry_msgs/PoseWithCovarianceStamped)
  /icp/pose        (geometry_msgs/PoseWithCovarianceStamped)
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    Quaternion, TransformStamped, PoseWithCovarianceStamped
)
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import TransformBroadcaster


# ── helpers ────────────────────────────────────────────────────────────────────

def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


# ── source state machine ───────────────────────────────────────────────────────

class SourceState:
    PREDICT_ONLY = 'PREDICT_ONLY'   # no external correction
    MCL_ACTIVE   = 'MCL_ACTIVE'     # localisation mode
    ICP_ACTIVE   = 'ICP_ACTIVE'     # mapping mode
    MCL_PRIORITY = 'MCL_PRIORITY'   # both alive — MCL wins


# ── EKF node ───────────────────────────────────────────────────────────────────

class PuzzlebotOdometry(Node):

    def __init__(self):
        super().__init__('puzzlebot_odom_ekf_node')

        # ── parameters ────────────────────────────────────────────────────
        self.declare_parameter('wheel_radius',   0.05)
        self.declare_parameter('wheel_base',     0.19)
        self.declare_parameter('rate',           50.0)
        self.declare_parameter('q_xy',           0.005)
        self.declare_parameter('q_theta',        0.01)
        self.declare_parameter('source_timeout', 0.5)
        self.declare_parameter('r_pos_default',  0.1)
        self.declare_parameter('r_yaw_default',  0.05)
        self.declare_parameter('max_innov_pos',  1.0)
        self.declare_parameter('max_innov_yaw',  1.5)

        self._R_wheel  = self.get_parameter('wheel_radius').value
        self._L        = self.get_parameter('wheel_base').value
        self._rate     = self.get_parameter('rate').value
        q_xy           = self.get_parameter('q_xy').value
        q_th           = self.get_parameter('q_theta').value
        self._timeout  = self.get_parameter('source_timeout').value
        self._r_pos    = self.get_parameter('r_pos_default').value
        self._r_yaw    = self.get_parameter('r_yaw_default').value
        self._max_ipos = self.get_parameter('max_innov_pos').value
        self._max_iyaw = self.get_parameter('max_innov_yaw').value

        self._Q_diag = np.array([q_xy, q_xy, q_th])

        # ── EKF state ─────────────────────────────────────────────────────
        self._x  = np.zeros(3)                   # [x, y, theta]
        self._P  = np.diag([1e-6, 1e-6, 1e-6])  # 3×3 covariance
        self._I3 = np.eye(3)

        # ── wheel velocities ──────────────────────────────────────────────
        self._wl = 0.0
        self._wr = 0.0

        # ── source tracking ───────────────────────────────────────────────
        # Wall-clock timestamps of the last message from each source.
        # -1.0 means "never received".
        self._last_mcl_t: float = -1.0
        self._last_icp_t: float = -1.0

        self._current_source = SourceState.PREDICT_ONLY
        self._prev_source    = SourceState.PREDICT_ONLY

        # ── lock ──────────────────────────────────────────────────────────
        self._ekf_lock = threading.Lock()

        # ── QoS ───────────────────────────────────────────────────────────
        enc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── subscribers ───────────────────────────────────────────────────
        self.create_subscription(Float32, '/VelocityEncL', self._cb_enc_l, enc_qos)
        self.create_subscription(Float32, '/VelocityEncR', self._cb_enc_r, enc_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, '/mcl/pose', self._cb_mcl, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/icp/pose', self._cb_icp, 10)

        # ── publishers ────────────────────────────────────────────────────
        self._pub_odom   = self.create_publisher(Odometry, '/odom', 10)
        self._pub_source = self.create_publisher(String,   '/ekf/active_source', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── timers ────────────────────────────────────────────────────────
        self._last_time = self.get_clock().now()
        self.create_timer(1.0 / self._rate, self._predict_and_publish)
        self.create_timer(0.2, self._watchdog)   # 5 Hz source health check

        self.get_logger().info(
            f'EKF Odometry started — '
            f'r={self._R_wheel} m, L={self._L} m, '
            f'rate={self._rate} Hz, timeout={self._timeout} s\n'
            f'Waiting for /mcl/pose or /icp/pose ...'
        )

    # ── encoder callbacks ──────────────────────────────────────────────────────

    def _cb_enc_l(self, msg: Float32): self._wl = msg.data
    def _cb_enc_r(self, msg: Float32): self._wr = msg.data

    # ── measurement callbacks ──────────────────────────────────────────────────

    def _cb_mcl(self, msg: PoseWithCovarianceStamped):
        """MCL pose arrived — stamp it, re-evaluate state, fuse if selected."""
        self._last_mcl_t = time.monotonic()
        self._update_source_state()

        if self._current_source in (SourceState.MCL_ACTIVE,
                                    SourceState.MCL_PRIORITY):
            self._ekf_update(msg, 'MCL')

    def _cb_icp(self, msg: PoseWithCovarianceStamped):
        """ICP pose arrived — stamp it, re-evaluate state, fuse if selected."""
        self._last_icp_t = time.monotonic()
        self._update_source_state()

        if self._current_source == SourceState.ICP_ACTIVE:
            self._ekf_update(msg, 'ICP')

    # ── source state machine ───────────────────────────────────────────────────

    def _update_source_state(self):
        """
        Decide which source is active based on recency of last messages.

        Priority rules:
          1. MCL beats ICP when both are alive (MCL is more accurate on a
             known map and should not be overridden by noisy scan matching).
          2. Whichever single source is alive wins.
          3. Neither alive → PREDICT_ONLY (pure dead-reckoning).
        """
        now    = time.monotonic()
        mcl_ok = (self._last_mcl_t > 0 and
                  now - self._last_mcl_t < self._timeout)
        icp_ok = (self._last_icp_t > 0 and
                  now - self._last_icp_t < self._timeout)

        if mcl_ok and icp_ok:
            new_state = SourceState.MCL_PRIORITY
        elif mcl_ok:
            new_state = SourceState.MCL_ACTIVE
        elif icp_ok:
            new_state = SourceState.ICP_ACTIVE
        else:
            new_state = SourceState.PREDICT_ONLY

        self._current_source = new_state
        self._log_source_change()

    def _watchdog(self):
        """
        5 Hz timer — catches timeouts that occur when messages simply stop
        arriving (e.g. a node crashes between publishes).
        """
        self._update_source_state()

    def _log_source_change(self):
        """Emit a log line and publish to /ekf/active_source only on change."""
        if self._current_source == self._prev_source:
            return
        self._prev_source = self._current_source

        labels = {
            SourceState.PREDICT_ONLY: '⚠  PREDICT_ONLY  — no external correction (covariance growing)',
            SourceState.MCL_ACTIVE:   '✓  MCL_ACTIVE    — localisation mode (/mcl/pose)',
            SourceState.ICP_ACTIVE:   '✓  ICP_ACTIVE    — mapping mode (/icp/pose)',
            SourceState.MCL_PRIORITY: '✓  MCL_PRIORITY  — both alive, MCL selected',
        }
        label = labels[self._current_source]

        if self._current_source == SourceState.PREDICT_ONLY:
            self.get_logger().warn(f'EKF source → {label}')
        else:
            self.get_logger().info(f'EKF source → {label}')

        msg = String()
        msg.data = self._current_source
        self._pub_source.publish(msg)

    # ── EKF PREDICT ───────────────────────────────────────────────────────────

    def _predict_and_publish(self):
        """
        EKF predict step — runs at `rate` Hz regardless of source state.

        x̂⁻  = f(x̂, u)
        P⁻   = F @ P @ Fᵀ + Q

        F = [[1, 0, -v·sin(θ)·dt],
             [0, 1,  v·cos(θ)·dt],
             [0, 0,  1          ]]
        """
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        if dt < 0.001 or dt > 0.5:
            self._last_time = now
            return
        self._last_time = now

        # Differential drive kinematics (sign convention from original node)
        v = self._R_wheel * (self._wr + self._wl) / 2.0
        w = self._R_wheel * (self._wl - self._wr) / self._L

        with self._ekf_lock:
            th = self._x[2]

            # State prediction
            self._x[0] += v * math.cos(th) * dt
            self._x[1] += v * math.sin(th) * dt
            self._x[2]  = normalize_angle(th + w * dt)

            # Covariance prediction
            F = np.array([
                [1.0, 0.0, -v * math.sin(th) * dt],
                [0.0, 1.0,  v * math.cos(th) * dt],
                [0.0, 0.0,  1.0],
            ])
            speed_scale = max(abs(v), abs(w), 0.01)
            Q = np.diag(self._Q_diag * dt * speed_scale)
            self._P = F @ self._P @ F.T + Q

            x_snap = self._x.copy()
            P_snap = self._P.copy()

        self._publish(x_snap, P_snap, v, w, now)

    # ── EKF UPDATE ────────────────────────────────────────────────────────────

    def _ekf_update(self, msg: PoseWithCovarianceStamped, source: str):
        """
        EKF measurement update for a direct pose observation (H = I).

        y  = z − x̂
        S  = P + R          (because H = I)
        K  = P @ S⁻¹
        x̂  = x̂ + K @ y
        P  = (I−K)P(I−K)ᵀ + KRKᵀ   Joseph form — numerically stable
        P  = (P+Pᵀ)/2                symmetry enforcement
        """
        z_x   = msg.pose.pose.position.x
        z_y   = msg.pose.pose.position.y
        z_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        # Build R from the covariance field of the incoming message.
        # Both MCL and ICP nodes fill this correctly; fall back to defaults
        # if the sender left it zero.
        cov  = msg.pose.covariance
        r_xx = cov[0]  if cov[0]  > 1e-9 else self._r_pos
        r_yy = cov[7]  if cov[7]  > 1e-9 else self._r_pos
        r_tt = cov[35] if cov[35] > 1e-9 else self._r_yaw
        R    = np.diag([r_xx, r_yy, r_tt])

        with self._ekf_lock:
            y    = np.array([z_x   - self._x[0],
                             z_y   - self._x[1],
                             normalize_angle(z_yaw - self._x[2])])

            # Innovation gate — reject implausibly large corrections
            if (math.hypot(y[0], y[1]) > self._max_ipos or
                    abs(y[2]) > self._max_iyaw):
                self.get_logger().warn(
                    f'[{source}] update REJECTED — '
                    f'Δpos={math.hypot(y[0],y[1]):.3f} m, '
                    f'Δyaw={math.degrees(y[2]):.1f}°'
                )
                return

            # Kalman gain (H = I simplifies S = H P Hᵀ + R to P + R)
            S = self._P + R
            K = self._P @ np.linalg.inv(S)

            # State update
            self._x    = self._x + K @ y
            self._x[2] = normalize_angle(self._x[2])

            # Covariance update — Joseph form
            IK      = self._I3 - K
            self._P = IK @ self._P @ IK.T + K @ R @ K.T
            self._P = (self._P + self._P.T) / 2.0   # enforce symmetry

        self.get_logger().debug(
            f'[{source}] EKF update — '
            f'Δpos={math.hypot(y[0],y[1]):.4f} m, '
            f'Δyaw={math.degrees(y[2]):.2f}°'
        )

    # ── publish helpers ────────────────────────────────────────────────────────

    def _publish(self, x: np.ndarray, P: np.ndarray,
                 v: float, w: float, now):
        odom = Odometry()
        odom.header.stamp    = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        odom.pose.pose.position.x  = x[0]
        odom.pose.pose.position.y  = x[1]
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = euler_to_quaternion(0.0, 0.0, x[2])

        c = [0.0] * 36
        c[0]  = P[0, 0]; c[1]  = P[0, 1]; c[5]  = P[0, 2]
        c[6]  = P[1, 0]; c[7]  = P[1, 1]; c[11] = P[1, 2]
        c[30] = P[2, 0]; c[31] = P[2, 1]; c[35] = P[2, 2]
        odom.pose.covariance = c

        odom.twist.twist.linear.x  = v
        odom.twist.twist.angular.z = w
        tc = [0.0] * 36
        tc[0]  = max(self._Q_diag[0], 0.001)
        tc[35] = max(self._Q_diag[2], 0.001)
        odom.twist.covariance = tc

        self._pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp    = now.to_msg()
        tf.header.frame_id = 'odom'
        tf.child_frame_id  = 'base_link'
        tf.transform.translation.x = x[0]
        tf.transform.translation.y = x[1]
        tf.transform.translation.z = 0.0
        tf.transform.rotation      = euler_to_quaternion(0.0, 0.0, x[2])
        self._tf_broadcaster.sendTransform(tf)


# ── entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()