#!/usr/bin/env python3
"""
puzzlebotOdometry.py — Full EKF Odometry with automatic MCL / ICP switching
=================================================================================

Measurement source selection
-----------------------------
Two external pose sources are supported:

  SOURCE A — /mcl/pose   (PoseWithCovarianceStamped)
      Published by puzzlebotMCL.py when localising on a known map.

  SOURCE B — /icp/pose   (PoseWithCovarianceStamped)
      Published by slam_node when building a map with ICP scan matching.

  SOURCE C — /aruco/pose (PoseWithCovarianceStamped)
      Published by aruco_localizer.py — triangulación de marcadores ArUco.
      Tiene prioridad más baja que MCL pero más alta que ICP.
      Útil cuando no hay mapa pero sí marcadores con posición conocida.

The node monitors BOTH topics.  Whichever one has published a message
within the last `source_timeout` seconds is considered **active**.
If both are active simultaneously, MCL takes priority (it is the more
accurate source once a map exists).

Source switching is automatic and logged clearly so you can see in the
terminal which source the EKF is currently fusing.

State machine
-------------
  PREDICT_ONLY  → no external source alive
  MCL_ACTIVE    → /mcl/pose   alive  (localisation mode)
  ICP_ACTIVE    → /icp/pose   alive  (mapping mode)
  ARUCO_ACTIVE  → /aruco/pose alive  (ArUco triangulation mode)
  MCL_PRIORITY  → both MCL + any other alive  (MCL wins)
  ARUCO_PRIORITY→ ARUCO + ICP alive            (ARUCO wins over ICP)

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
  wheel_radius        0.05   [m]
  wheel_base          0.19   [m]
  rate                50.0   [Hz]
  q_xy                0.005  process noise – translation  [m²/s]
  q_theta             0.01   process noise – rotation     [rad²/s]
  source_timeout      0.5    seconds of silence before MCL/ARUCO declared dead
  icp_source_timeout  2.5    seconds of silence before ICP declared dead
                             (ICP publishes only on keyframes, not every scan)
  r_pos_default       0.1    fallback R diagonal for xy   [m²]
  r_yaw_default       0.05   fallback R diagonal for yaw  [rad²]
  max_innov_pos       1.0    innovation gate – position   [m]
  max_innov_yaw       1.5    innovation gate – yaw        [rad]

Publishes
---------
  /odom                  (nav_msgs/Odometry)
  /ekf/active_source     (std_msgs/String)   current source name for monitoring
  TF: odom → base_link   (raw dead-reckoning pose)
  TF: map  → odom        (correction offset; identity until first EKF update)

Subscribes
----------
  /VelocityEncL    (std_msgs/Float32)
  /VelocityEncR    (std_msgs/Float32)
  /mcl/pose        (geometry_msgs/PoseWithCovarianceStamped)
  /icp/pose        (geometry_msgs/PoseWithCovarianceStamped)
  /aruco/pose      (geometry_msgs/PoseWithCovarianceStamped)

Fixes (v2)
----------
  FIX1 — Angular velocity sign corrected: w = R*(wr - wl)/L  (was wl - wr)
  FIX2 — Wheel velocity read inside ekf_lock to prevent data race with
          encoder callbacks running on separate executor threads.
  FIX3 — Process noise Q = diag(q_diag)*dt only; removed erroneous
          speed_scale multiplier that inflated covariance at high speed.
  FIX4 — dt upper-bound tightened from 0.5 s to 0.08 s to avoid integrating
          stale velocities after a long pause.
  FIX5 — Innovation gate bypassed for the very first update of each source
          so large initial offsets (common on startup) are not silently dropped.
  FIX6 — Separate icp_source_timeout parameter (default 2.5 s) because ICP
          only publishes on accepted keyframes, not every scan cycle.
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
    PREDICT_ONLY   = 'PREDICT_ONLY'
    MCL_ACTIVE     = 'MCL_ACTIVE'
    ICP_ACTIVE     = 'ICP_ACTIVE'
    ARUCO_ACTIVE   = 'ARUCO_ACTIVE'
    MCL_PRIORITY   = 'MCL_PRIORITY'
    ARUCO_PRIORITY = 'ARUCO_PRIORITY'


# ── EKF node ───────────────────────────────────────────────────────────────────

class PuzzlebotOdometry(Node):

    def __init__(self):
        super().__init__('puzzlebot_odom_ekf_node')

        # ── parameters ────────────────────────────────────────────────────
        self.declare_parameter('initial_yaw',         0.0)
        self.declare_parameter('wheel_radius',        0.05)
        self.declare_parameter('wheel_base',          0.19)
        self.declare_parameter('rate',                50.0)
        self.declare_parameter('q_xy',                0.005)
        self.declare_parameter('q_theta',             0.01)
        self.declare_parameter('source_timeout',      1.5)   # FIX9: was 0.5 — 5 frames at 10 Hz is too tight
        # FIX6: separate, longer timeout for ICP (keyframe-gated, not every scan)
        self.declare_parameter('icp_source_timeout',  2.5)
        self.declare_parameter('r_pos_default',       0.1)
        self.declare_parameter('r_yaw_default',       0.05)
        self.declare_parameter('max_innov_pos',       1.0)
        self.declare_parameter('max_innov_yaw',       1.5)

        self._R_wheel    = self.get_parameter('wheel_radius').value
        self._L          = self.get_parameter('wheel_base').value
        self._rate       = self.get_parameter('rate').value
        q_xy             = self.get_parameter('q_xy').value
        q_th             = self.get_parameter('q_theta').value
        self._timeout    = self.get_parameter('source_timeout').value
        self._icp_timeout = self.get_parameter('icp_source_timeout').value  # FIX6
        self._r_pos      = self.get_parameter('r_pos_default').value
        self._r_yaw      = self.get_parameter('r_yaw_default').value
        self._max_ipos   = self.get_parameter('max_innov_pos').value
        self._max_iyaw   = self.get_parameter('max_innov_yaw').value

        self._Q_diag = np.array([q_xy, q_xy, q_th])

        # ── EKF state (corrected pose — map frame) ────────────────────────
        self._x  = np.zeros(3)
        self._x[2] = self.get_parameter('initial_yaw').value
        self._P  = np.diag([1e-6, 1e-6, 1e-6])
        self._I3 = np.eye(3)

        # ── Raw dead-reckoning pose (odom frame, never corrected) ─────────
        # Tracks pure wheel integration so we can compute the map->odom offset
        # as:  map_T_base - odom_T_base
        self._raw_x  = np.zeros(3)
        self._raw_x[2] = self.get_parameter('initial_yaw').value

        # ── map -> odom offset (updated on every EKF measurement update) ──
        # Stored as [tx, ty, yaw] broadcasted by _publish().
        # Identity until the first ArUco/MCL/ICP fix.
        self._map_odom = np.zeros(3)   # [tx, ty, dyaw]

        # ── wheel velocities ──────────────────────────────────────────────
        self._wl = 0.0
        self._wr = 0.0

        # ── source tracking ───────────────────────────────────────────────
        self._last_mcl_t:   float = -1.0
        self._last_icp_t:   float = -1.0
        self._last_aruco_t: float = -1.0

        # FIX5: track whether each source has ever been fused (bypass gate on first update)
        self._mcl_initialised   = False
        self._icp_initialised   = False
        self._aruco_initialised = False

        self._current_source = SourceState.PREDICT_ONLY
        self._prev_source    = SourceState.PREDICT_ONLY

        # ── lock ──────────────────────────────────────────────────────────
        # Single lock guards: EKF state (_x, _P), wheel velocities (_wl, _wr),
        # and source timestamps / initialisation flags.
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
            PoseWithCovarianceStamped, '/mcl/pose',   self._cb_mcl,   10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/icp/pose',   self._cb_icp,   10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/aruco/pose', self._cb_aruco, 10)

        # ── publishers ────────────────────────────────────────────────────
        self._pub_odom   = self.create_publisher(Odometry, '/odom', 10)
        self._pub_source = self.create_publisher(String,   '/ekf/active_source', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── timers ────────────────────────────────────────────────────────
        self._last_time = self.get_clock().now()
        self.create_timer(1.0 / self._rate, self._predict_and_publish)
        self.create_timer(0.2, self._watchdog)

        self.get_logger().info(
            f'EKF Odometry started — '
            f'r={self._R_wheel} m, L={self._L} m, '
            f'rate={self._rate} Hz, timeout={self._timeout} s, '
            f'icp_timeout={self._icp_timeout} s\n'
            f'Waiting for /mcl/pose, /icp/pose or /aruco/pose ...'
        )

    # ── encoder callbacks ──────────────────────────────────────────────────────

    def _cb_enc_l(self, msg: Float32):
        # FIX2: protect wheel velocity writes with the EKF lock
        with self._ekf_lock:
            self._wl = msg.data

    def _cb_enc_r(self, msg: Float32):
        # FIX2: protect wheel velocity writes with the EKF lock
        with self._ekf_lock:
            self._wr = msg.data

    # ── measurement callbacks ──────────────────────────────────────────────────

    def _cb_mcl(self, msg: PoseWithCovarianceStamped):
        with self._ekf_lock:
            self._last_mcl_t = time.monotonic()
        self._update_source_state()
        # MCL solo actualiza si ArUco no está disponible (ArUco tiene prioridad)
        if self._current_source in (SourceState.MCL_ACTIVE,
                                    SourceState.MCL_PRIORITY):
            self._ekf_update(msg, 'MCL')

    def _cb_icp(self, msg: PoseWithCovarianceStamped):
        with self._ekf_lock:
            self._last_icp_t = time.monotonic()
        self._update_source_state()
        # ICP solo actualiza si ni ArUco ni MCL están disponibles
        if self._current_source == SourceState.ICP_ACTIVE:
            self._ekf_update(msg, 'ICP')

    def _cb_aruco(self, msg: PoseWithCovarianceStamped):
        with self._ekf_lock:
            self._last_aruco_t = time.monotonic()
        self._update_source_state()
        # ArUco siempre actualiza cuando está activo (prioridad máxima)
        if self._current_source in (SourceState.ARUCO_ACTIVE,
                                    SourceState.ARUCO_PRIORITY):
            self._ekf_update(msg, 'ARUCO')

    # ── source state machine ───────────────────────────────────────────────────

    def _update_source_state(self):
        now = time.monotonic()
        with self._ekf_lock:
            last_mcl   = self._last_mcl_t
            last_icp   = self._last_icp_t
            last_aruco = self._last_aruco_t

        mcl_ok   = (last_mcl   > 0 and now - last_mcl   < self._timeout)
        # FIX6: ICP uses its own (longer) timeout
        icp_ok   = (last_icp   > 0 and now - last_icp   < self._icp_timeout)
        aruco_ok = (last_aruco > 0 and now - last_aruco < self._timeout)

        # Prioridad: ArUco (más preciso cuando hay marcadores) > MCL > ICP
        if aruco_ok and mcl_ok:
            new_state = SourceState.ARUCO_PRIORITY   # ArUco gana sobre MCL
        elif aruco_ok and icp_ok:
            new_state = SourceState.ARUCO_PRIORITY   # ArUco gana sobre ICP
        elif aruco_ok:
            new_state = SourceState.ARUCO_ACTIVE
        elif mcl_ok:
            new_state = SourceState.MCL_PRIORITY if icp_ok \
                        else SourceState.MCL_ACTIVE
        elif icp_ok:
            new_state = SourceState.ICP_ACTIVE
        else:
            new_state = SourceState.PREDICT_ONLY

        self._current_source = new_state
        self._log_source_change()

    def _watchdog(self):
        self._update_source_state()

    def _log_source_change(self):
        if self._current_source == self._prev_source:
            return
        self._prev_source = self._current_source

        labels = {
            SourceState.PREDICT_ONLY:   '⚠  PREDICT_ONLY   — no external correction (covariance growing)',
            SourceState.MCL_ACTIVE:     '✓  MCL_ACTIVE     — localisation mode (/mcl/pose)',
            SourceState.ICP_ACTIVE:     '✓  ICP_ACTIVE     — mapping mode (/icp/pose)',
            SourceState.ARUCO_ACTIVE:   '✓  ARUCO_ACTIVE   — ArUco triangulation (/aruco/pose)',
            SourceState.MCL_PRIORITY:   '✓  MCL_PRIORITY   — MCL selected (beats ARUCO/ICP)',
            SourceState.ARUCO_PRIORITY: '✓  ARUCO_PRIORITY — ARUCO selected (beats ICP)',
        }
        label = labels[self._current_source]

        if self._current_source == SourceState.PREDICT_ONLY:
            self.get_logger().warn(f'EKF source → {label}')
            # FIX8: reset initialisation flags so the next re-acquisition from
            # any source bypasses the innovation gate (same as first-ever update).
            # Without this, drift accumulated during dead-reckoning causes the
            # first correction after re-acquisition to be wrongly rejected.
            with self._ekf_lock:
                self._aruco_initialised = False
                self._mcl_initialised   = False
                self._icp_initialised   = False
        else:
            self.get_logger().info(f'EKF source → {label}')

        msg = String()
        msg.data = self._current_source
        self._pub_source.publish(msg)

    # ── EKF PREDICT ───────────────────────────────────────────────────────────

    def _predict_and_publish(self):
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        # FIX4: tightened upper bound — 0.08 s (~4 cycles) instead of 0.5 s
        if dt < 0.001 or dt > 0.08:
            self._last_time = now
            return
        self._last_time = now

        with self._ekf_lock:
            # FIX2: read wheel velocities inside the lock
            wl = self._wl
            wr = self._wr

            # FIX1: correct sign — standard differential drive is (wr - wl)/L
            v = self._R_wheel * (wr + wl) / 2.0
            w = self._R_wheel * (wr - wl) / self._L

            th = self._x[2]

            # Corrected EKF state prediction (map frame)
            self._x[0] += v * math.cos(th) * dt
            self._x[1] += v * math.sin(th) * dt
            self._x[2]  = normalize_angle(th + w * dt)

            # Raw dead-reckoning prediction (odom frame — never corrected)
            raw_th = self._raw_x[2]
            self._raw_x[0] += v * math.cos(raw_th) * dt
            self._raw_x[1] += v * math.sin(raw_th) * dt
            self._raw_x[2]  = normalize_angle(raw_th + w * dt)

            # Covariance prediction
            # FIX3: Q = diag(q_diag) * dt only — no speed_scale multiplier
            F = np.array([
                [1.0, 0.0, -v * math.sin(th) * dt],
                [0.0, 1.0,  v * math.cos(th) * dt],
                [0.0, 0.0,  1.0],
            ])
            Q = np.diag(self._Q_diag * dt)
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

        cov  = msg.pose.covariance
        r_xx = cov[0]  if cov[0]  > 1e-9 else self._r_pos
        r_yy = cov[7]  if cov[7]  > 1e-9 else self._r_pos
        r_tt = cov[35] if cov[35] > 1e-9 else self._r_yaw
        R    = np.diag([r_xx, r_yy, r_tt])

        with self._ekf_lock:
            # FIX5: determine whether this source has been initialised yet
            if source == 'MCL':
                first_update = not self._mcl_initialised
                self._mcl_initialised = True
            elif source == 'ICP':
                first_update = not self._icp_initialised
                self._icp_initialised = True
            else:
                first_update = not self._aruco_initialised
                self._aruco_initialised = True

            y = np.array([z_x   - self._x[0],
                          z_y   - self._x[1],
                          normalize_angle(z_yaw - self._x[2])])

            # FIX5: bypass innovation gate on the very first update from a
            #        source so large startup offsets are not silently dropped.
            # FIX7: adaptive gate — when covariance is large (long PREDICT_ONLY
            #        stretch) the gate widens to 3σ so real corrections aren't
            #        rejected precisely when they are most needed.
            if not first_update:
                pos_sigma = math.sqrt(max(self._P[0, 0] + self._P[1, 1], 0.0))
                yaw_sigma = math.sqrt(max(self._P[2, 2], 0.0))
                gate_pos  = max(self._max_ipos, 3.0 * pos_sigma)
                gate_yaw  = max(self._max_iyaw, 3.0 * yaw_sigma)
                if (math.hypot(y[0], y[1]) > gate_pos or
                        abs(y[2]) > gate_yaw):
                    self.get_logger().warn(
                        f'[{source}] update REJECTED — '
                        f'Δpos={math.hypot(y[0],y[1]):.3f} m (gate={gate_pos:.2f}), '
                        f'Δyaw={math.degrees(y[2]):.1f}° (gate={math.degrees(gate_yaw):.1f}°)'
                    )
                    return

            S = self._P + R
            K = self._P @ np.linalg.inv(S)

            self._x    = self._x + K @ y
            self._x[2] = normalize_angle(self._x[2])

            IK      = self._I3 - K
            self._P = IK @ self._P @ IK.T + K @ R @ K.T
            self._P = (self._P + self._P.T) / 2.0

            # ── Recompute map -> odom offset ──────────────────────────────
            # After the EKF correction, self._x is the best estimate of
            # base_link in the MAP frame.  self._raw_x is where pure wheel
            # odometry places base_link in the ODOM frame.
            #
            # We want map_T_odom such that:
            #   map_T_base = map_T_odom * odom_T_base
            #
            # With planar 2-D transforms:
            #   map_T_odom.yaw = x_map.yaw - raw_x.yaw
            #   map_T_odom.pos = x_map.pos - R(map_T_odom.yaw) * raw_x.pos
            #
            dyaw = normalize_angle(self._x[2] - self._raw_x[2])
            cos_d = math.cos(dyaw)
            sin_d = math.sin(dyaw)
            # Rotate raw odom position into map frame then subtract
            rx = self._raw_x[0]
            ry = self._raw_x[1]
            self._map_odom[0] = self._x[0] - (cos_d * rx - sin_d * ry)
            self._map_odom[1] = self._x[1] - (sin_d * rx + cos_d * ry)
            self._map_odom[2] = dyaw

        self.get_logger().debug(
            f'[{source}]{"(init)" if first_update else ""} EKF update — '
            f'Δpos={math.hypot(y[0],y[1]):.4f} m, '
            f'Δyaw={math.degrees(y[2]):.2f}°'
        )

    # ── publish helpers ────────────────────────────────────────────────────────

    def _publish(self, x: np.ndarray, P: np.ndarray,
                 v: float, w: float, now):
        odom = Odometry()
        odom.header.stamp    = now.to_msg()
        # /odom carries the EKF-corrected pose in the MAP frame so that
        # consumers (A*, go-to-goal) always see the best estimate.
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
        # Fixed: twist covariance uses physically meaningful velocity uncertainty,
        # not the process-noise q_xy which has different units/semantics.
        tc = [0.0] * 36
        tc[0]  = 0.01   # linear velocity uncertainty (m/s)²
        tc[35] = 0.01   # angular velocity uncertainty (rad/s)²
        odom.twist.covariance = tc

        self._pub_odom.publish(odom)

        # ── TF: odom → base_link (raw dead-reckoning, never corrected) ────
        with self._ekf_lock:
            raw = self._raw_x.copy()
            map_odom = self._map_odom.copy()

        tf_ob = TransformStamped()
        tf_ob.header.stamp    = now.to_msg()
        tf_ob.header.frame_id = 'odom'
        tf_ob.child_frame_id  = 'base_link'
        tf_ob.transform.translation.x = raw[0]
        tf_ob.transform.translation.y = raw[1]
        tf_ob.transform.translation.z = 0.0
        tf_ob.transform.rotation      = euler_to_quaternion(0.0, 0.0, raw[2])
        self._tf_broadcaster.sendTransform(tf_ob)

        # ── TF: map → odom (EKF correction offset) ────────────────────────
        # Identity until the first measurement update fires.
        tf_mo = TransformStamped()
        tf_mo.header.stamp    = now.to_msg()
        tf_mo.header.frame_id = 'map'
        tf_mo.child_frame_id  = 'odom'
        tf_mo.transform.translation.x = map_odom[0]
        tf_mo.transform.translation.y = map_odom[1]
        tf_mo.transform.translation.z = 0.0
        tf_mo.transform.rotation      = euler_to_quaternion(0.0, 0.0, map_odom[2])
        self._tf_broadcaster.sendTransform(tf_mo)


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