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

EKF mode selection  (parámetro: ekf_mode)
-----------------------------------------
  odometry_only  → solo encoders, sin fuente externa (dead-reckoning puro)
  aruco          → encoders + /aruco/pose
  mcl            → encoders + /mcl/pose
  icp            → encoders + /icp/pose
  full           → todas las fuentes con la lógica de prioridad completa
                   (MCL > ARUCO > ICP)   ← comportamiento original

  Ejemplo de uso en launch:
      parameters=[{'ekf_mode': 'aruco'}]

  Ejemplo desde CLI:
      ros2 run iolair odometry --ros-args -p ekf_mode:=mcl

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
  ekf_mode            'full'  modo de fusión:
                              odometry_only | aruco | mcl | icp | full
  wheel_radius        0.05   [m]
  wheel_base          0.19   [m]
  rate                50.0   [Hz]
  q_xy                0.005  process noise – translation  [m²/s]
  q_theta             0.01   process noise – rotation     [rad²/s]
  source_timeout      0.5    seconds of silence before MCL/ARUCO declared dead
  icp_source_timeout  2.5    seconds of silence before ICP declared dead
  r_pos_default       0.1    fallback R diagonal for xy   [m²]
  r_yaw_default       0.05   fallback R diagonal for yaw  [rad²]
  max_innov_pos       1.0    innovation gate – position   [m]
  max_innov_yaw       1.5    innovation gate – yaw        [rad]

Publishes
---------
  /odom                  (nav_msgs/Odometry)
  /ekf/active_source     (std_msgs/String)   current source name for monitoring
  TF: odom → base_link

Subscribes
----------
  /VelocityEncL    (std_msgs/Float32)
  /VelocityEncR    (std_msgs/Float32)
  /mcl/pose        (geometry_msgs/PoseWithCovarianceStamped)  si el modo lo permite
  /icp/pose        (geometry_msgs/PoseWithCovarianceStamped)  si el modo lo permite
  /aruco/pose      (geometry_msgs/PoseWithCovarianceStamped)  si el modo lo permite
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


# ── valid modes ────────────────────────────────────────────────────────────────

VALID_MODES = {'odometry_only', 'aruco', 'mcl', 'icp', 'full'}


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
        self.declare_parameter('ekf_mode',            'full')
        self.declare_parameter('initial_yaw',         0.0)
        self.declare_parameter('wheel_radius',        0.05)
        self.declare_parameter('wheel_base',          0.19)
        self.declare_parameter('rate',                50.0)
        self.declare_parameter('q_xy',                0.005)
        self.declare_parameter('q_theta',             0.01)
        self.declare_parameter('source_timeout',      0.5)
        self.declare_parameter('icp_source_timeout',  2.5)
        self.declare_parameter('r_pos_default',       0.1)
        self.declare_parameter('r_yaw_default',       0.05)
        self.declare_parameter('max_innov_pos',       1.0)
        self.declare_parameter('max_innov_yaw',       1.5)

        # ── ekf_mode validation ───────────────────────────────────────────
        raw_mode = self.get_parameter('ekf_mode').value.strip().lower()
        if raw_mode not in VALID_MODES:
            self.get_logger().warn(
                f'ekf_mode="{raw_mode}" no reconocido. '
                f'Opciones válidas: {sorted(VALID_MODES)}. '
                f'Usando "full" por defecto.'
            )
            raw_mode = 'full'
        self._ekf_mode = raw_mode

        # Flags de qué fuentes están habilitadas en este modo
        self._use_mcl   = self._ekf_mode in ('mcl',   'full')
        self._use_icp   = self._ekf_mode in ('icp',   'full')
        self._use_aruco = self._ekf_mode in ('aruco', 'full')

        self._R_wheel     = self.get_parameter('wheel_radius').value
        self._L           = self.get_parameter('wheel_base').value
        self._rate        = self.get_parameter('rate').value
        q_xy              = self.get_parameter('q_xy').value
        q_th              = self.get_parameter('q_theta').value
        self._timeout     = self.get_parameter('source_timeout').value
        self._icp_timeout = self.get_parameter('icp_source_timeout').value
        self._r_pos       = self.get_parameter('r_pos_default').value
        self._r_yaw       = self.get_parameter('r_yaw_default').value
        self._max_ipos    = self.get_parameter('max_innov_pos').value
        self._max_iyaw    = self.get_parameter('max_innov_yaw').value

        self._Q_diag = np.array([q_xy, q_xy, q_th])

        # ── EKF state ─────────────────────────────────────────────────────
        self._x    = np.zeros(3)
        self._x[2] = self.get_parameter('initial_yaw').value
        self._P    = np.diag([1e-6, 1e-6, 1e-6])
        self._I3   = np.eye(3)

        # ── wheel velocities ──────────────────────────────────────────────
        self._wl = 0.0
        self._wr = 0.0

        # ── source tracking ───────────────────────────────────────────────
        self._last_mcl_t:   float = -1.0
        self._last_icp_t:   float = -1.0
        self._last_aruco_t: float = -1.0

        self._mcl_initialised   = False
        self._icp_initialised   = False
        self._aruco_initialised = False

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

        # ── subscribers (encoders siempre activos) ────────────────────────
        self.create_subscription(Float32, '/VelocityEncL', self._cb_enc_l, enc_qos)
        self.create_subscription(Float32, '/VelocityEncR', self._cb_enc_r, enc_qos)

        # Fuentes externas: solo se suscriben si el modo las habilita
        if self._use_mcl:
            self.create_subscription(
                PoseWithCovarianceStamped, '/mcl/pose', self._cb_mcl, 10)
        if self._use_icp:
            self.create_subscription(
                PoseWithCovarianceStamped, '/icp/pose', self._cb_icp, 10)
        if self._use_aruco:
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

        # ── log de modo activo ────────────────────────────────────────────
        sources_enabled = []
        if self._use_mcl:   sources_enabled.append('/mcl/pose')
        if self._use_aruco: sources_enabled.append('/aruco/pose')
        if self._use_icp:   sources_enabled.append('/icp/pose')
        sources_str = ', '.join(sources_enabled) if sources_enabled else 'ninguna (dead-reckoning)'

        self.get_logger().info(
            f'EKF Odometry iniciado\n'
            f'  ekf_mode  : {self._ekf_mode}\n'
            f'  Fuentes   : {sources_str}\n'
            f'  r={self._R_wheel} m, L={self._L} m, rate={self._rate} Hz'
        )

    # ── encoder callbacks ──────────────────────────────────────────────────────

    def _cb_enc_l(self, msg: Float32):
        with self._ekf_lock:
            self._wl = msg.data

    def _cb_enc_r(self, msg: Float32):
        with self._ekf_lock:
            self._wr = msg.data

    # ── measurement callbacks ──────────────────────────────────────────────────

    def _cb_mcl(self, msg: PoseWithCovarianceStamped):
        with self._ekf_lock:
            self._last_mcl_t = time.monotonic()
        self._update_source_state()
        if self._current_source in (SourceState.MCL_ACTIVE,
                                    SourceState.MCL_PRIORITY):
            self._ekf_update(msg, 'MCL')

    def _cb_icp(self, msg: PoseWithCovarianceStamped):
        with self._ekf_lock:
            self._last_icp_t = time.monotonic()
        self._update_source_state()
        if self._current_source == SourceState.ICP_ACTIVE:
            self._ekf_update(msg, 'ICP')

    def _cb_aruco(self, msg: PoseWithCovarianceStamped):
        with self._ekf_lock:
            self._last_aruco_t = time.monotonic()
        self._update_source_state()
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

        # Solo considerar fuentes que el modo habilita
        mcl_ok   = (self._use_mcl   and
                    last_mcl   > 0 and now - last_mcl   < self._timeout)
        icp_ok   = (self._use_icp   and
                    last_icp   > 0 and now - last_icp   < self._icp_timeout)
        aruco_ok = (self._use_aruco and
                    last_aruco > 0 and now - last_aruco < self._timeout)

        # Modo odometry_only: siempre PREDICT_ONLY
        if self._ekf_mode == 'odometry_only':
            new_state = SourceState.PREDICT_ONLY

        # Modos de fuente única: sin lógica de prioridad
        elif self._ekf_mode == 'mcl':
            new_state = SourceState.MCL_ACTIVE if mcl_ok \
                        else SourceState.PREDICT_ONLY

        elif self._ekf_mode == 'icp':
            new_state = SourceState.ICP_ACTIVE if icp_ok \
                        else SourceState.PREDICT_ONLY

        elif self._ekf_mode == 'aruco':
            new_state = SourceState.ARUCO_ACTIVE if aruco_ok \
                        else SourceState.PREDICT_ONLY

        # Modo full: lógica de prioridad completa (MCL > ARUCO > ICP)
        else:
            if mcl_ok:
                new_state = SourceState.MCL_PRIORITY if (icp_ok or aruco_ok) \
                            else SourceState.MCL_ACTIVE
            elif aruco_ok and icp_ok:
                new_state = SourceState.ARUCO_PRIORITY
            elif aruco_ok:
                new_state = SourceState.ARUCO_ACTIVE
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
            SourceState.PREDICT_ONLY:   '⚠  PREDICT_ONLY   — sin corrección externa (covarianza creciendo)',
            SourceState.MCL_ACTIVE:     '✓  MCL_ACTIVE     — modo localización (/mcl/pose)',
            SourceState.ICP_ACTIVE:     '✓  ICP_ACTIVE     — modo mapeo (/icp/pose)',
            SourceState.ARUCO_ACTIVE:   '✓  ARUCO_ACTIVE   — triangulación ArUco (/aruco/pose)',
            SourceState.MCL_PRIORITY:   '✓  MCL_PRIORITY   — MCL seleccionado (gana a ARUCO/ICP)',
            SourceState.ARUCO_PRIORITY: '✓  ARUCO_PRIORITY — ARUCO seleccionado (gana a ICP)',
        }
        label = labels[self._current_source]

        if self._current_source == SourceState.PREDICT_ONLY:
            self.get_logger().warn(
                f'[ekf_mode={self._ekf_mode}] EKF source → {label}')
        else:
            self.get_logger().info(
                f'[ekf_mode={self._ekf_mode}] EKF source → {label}')

        msg = String()
        msg.data = f'{self._ekf_mode}:{self._current_source}'
        self._pub_source.publish(msg)

    # ── EKF PREDICT ───────────────────────────────────────────────────────────

    def _predict_and_publish(self):
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        if dt < 0.001 or dt > 0.08:
            self._last_time = now
            return
        self._last_time = now

        with self._ekf_lock:
            wl = self._wl
            wr = self._wr

            v = self._R_wheel * (wr + wl) / 2.0
            w = self._R_wheel * (wr - wl) / self._L

            th = self._x[2]

            self._x[0] += v * math.cos(th) * dt
            self._x[1] += v * math.sin(th) * dt
            self._x[2]  = normalize_angle(th + w * dt)

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
        z_x   = msg.pose.pose.position.x
        z_y   = msg.pose.pose.position.y
        z_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        cov  = msg.pose.covariance
        r_xx = cov[0]  if cov[0]  > 1e-9 else self._r_pos
        r_yy = cov[7]  if cov[7]  > 1e-9 else self._r_pos
        r_tt = cov[35] if cov[35] > 1e-9 else self._r_yaw
        R    = np.diag([r_xx, r_yy, r_tt])

        with self._ekf_lock:
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

            if not first_update:
                if (math.hypot(y[0], y[1]) > self._max_ipos or
                        abs(y[2]) > self._max_iyaw):
                    self.get_logger().warn(
                        f'[{source}] update RECHAZADO — '
                        f'Δpos={math.hypot(y[0],y[1]):.3f} m, '
                        f'Δyaw={math.degrees(y[2]):.1f}°'
                    )
                    return

            S = self._P + R
            K = self._P @ np.linalg.inv(S)

            self._x    = self._x + K @ y
            self._x[2] = normalize_angle(self._x[2])

            IK      = self._I3 - K
            self._P = IK @ self._P @ IK.T + K @ R @ K.T
            self._P = (self._P + self._P.T) / 2.0

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
        tc[0]  = 0.01
        tc[35] = 0.01
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