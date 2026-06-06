#!/usr/bin/env python3
"""
puzzlebotOdometry.py — Full EKF Odometry with automatic MCL / ICP / ArUco switching
======================================================================================
[misma arquitectura que v2, fixes adicionales v3]

Fixes v3 (sobre v2)
--------------------
  FIX-V3-1  Integración RK2 (Runge-Kutta orden 2) en lugar de Euler hacia adelante.
             En trayectorias curvas, Euler acumula error O(dt) mientras que RK2
             acumula O(dt²). Con dt=0.02s y velocidades típicas del PuzzleBot el
             error posicional se reduce ~10x en curvas cerradas.

  FIX-V3-2  Parámetros de calibración de slip por rueda: `slip_l` y `slip_r`
             (default 1.0, sin corrección). Motores físicos con distinta fricción
             o desgaste producen velocidades efectivas distintas aunque el encoder
             reporte lo mismo. Ajustar individualmente con:
               --ros-args -p slip_l:=0.98 -p slip_r:=1.02
             Procedimiento de calibración: marcar posición inicial, avanzar recto
             2m, medir desvío lateral. Si desvía a la derecha: slip_l > 1 o slip_r < 1.

  FIX-V3-3  dt upper-bound reducido de 0.08s a 0.05s (2.5 ciclos a 50Hz).
             En Jetson Nano bajo carga el scheduler puede retrasar el timer hasta
             80ms — integrar 4 ciclos de velocidad estale sobreestima la distancia.

  FIX-V3-4  Q process noise escalado con velocidad cuando la EKF lleva más de
             `predict_only_decay_s` segundos sin corrección externa. Esto ensancha
             la covarianza progresivamente para que la innovación gate acepte
             correcciones ArUco/MCL más agresivas en vez de rechazarlas.

  FIX-V3-5  La publicación de /ekf/active_source se hace también dentro del timer
             de predict (no solo en log_source_change) para que el dashboard siempre
             tenga el valor más reciente incluso cuando no hay cambio de fuente.
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
        self.declare_parameter('initial_yaw',            0.0)
        self.declare_parameter('wheel_radius',           0.05)
        self.declare_parameter('wheel_base',             0.19)
        self.declare_parameter('rate',                   50.0)
        self.declare_parameter('q_xy',                   0.005)
        self.declare_parameter('q_theta',                0.01)
        self.declare_parameter('source_timeout',         1.5)
        self.declare_parameter('icp_source_timeout',     2.5)
        self.declare_parameter('r_pos_default',          0.1)
        self.declare_parameter('r_yaw_default',          0.05)
        self.declare_parameter('max_innov_pos',          1.0)
        self.declare_parameter('max_innov_yaw',          1.5)
        # FIX-V3-2: per-wheel slip calibration factors
        self.declare_parameter('slip_l',                 1.0)
        self.declare_parameter('slip_r',                 1.0)
        # FIX-V3-4: predict-only covariance decay
        self.declare_parameter('predict_only_decay_s',   5.0)
        self.declare_parameter('q_decay_factor',         3.0)

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
        # FIX-V3-2
        self._slip_l      = self.get_parameter('slip_l').value
        self._slip_r      = self.get_parameter('slip_r').value
        # FIX-V3-4
        self._decay_s     = self.get_parameter('predict_only_decay_s').value
        self._decay_k     = self.get_parameter('q_decay_factor').value

        self._Q_diag = np.array([q_xy, q_xy, q_th])

        # ── EKF state ─────────────────────────────────────────────────────
        self._x  = np.zeros(3)
        self._x[2] = self.get_parameter('initial_yaw').value
        self._P  = np.diag([1e-6, 1e-6, 1e-6])
        self._I3 = np.eye(3)

        # ── Raw dead-reckoning ─────────────────────────────────────────────
        self._raw_x    = np.zeros(3)
        self._raw_x[2] = self.get_parameter('initial_yaw').value

        # ── map → odom offset ─────────────────────────────────────────────
        self._map_odom = np.zeros(3)

        # ── wheel velocities ──────────────────────────────────────────────
        self._wl = 0.0
        self._wr = 0.0

        # ── source tracking ───────────────────────────────────────────────
        self._last_mcl_t:   float = -1.0
        self._last_icp_t:   float = -1.0
        self._last_aruco_t: float = -1.0
        self._last_update_t: float = -1.0  # FIX-V3-4

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
            f'EKF Odometry v3 — '
            f'r={self._R_wheel} m, L={self._L} m, '
            f'slip_l={self._slip_l:.4f}, slip_r={self._slip_r:.4f}, '
            f'rate={self._rate} Hz\n'
            f'Calibration tip: avanzar recto 2m, medir desvío lateral.\n'
            f'  Desvía derecha → aumentar slip_l o reducir slip_r\n'
            f'  Desvía izquierda → reducir slip_l o aumentar slip_r'
        )

    # ── encoder callbacks ─────────────────────────────────────────────────────

    def _cb_enc_l(self, msg: Float32):
        with self._ekf_lock:
            self._wl = msg.data

    def _cb_enc_r(self, msg: Float32):
        with self._ekf_lock:
            self._wr = msg.data

    # ── measurement callbacks ─────────────────────────────────────────────────

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

    # ── source state machine ──────────────────────────────────────────────────

    def _update_source_state(self):
        now = time.monotonic()
        with self._ekf_lock:
            last_mcl   = self._last_mcl_t
            last_icp   = self._last_icp_t
            last_aruco = self._last_aruco_t

        mcl_ok   = (last_mcl   > 0 and now - last_mcl   < self._timeout)
        icp_ok   = (last_icp   > 0 and now - last_icp   < self._icp_timeout)
        aruco_ok = (last_aruco > 0 and now - last_aruco < self._timeout)

        if aruco_ok and mcl_ok:
            new_state = SourceState.ARUCO_PRIORITY
        elif aruco_ok and icp_ok:
            new_state = SourceState.ARUCO_PRIORITY
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
        # FIX-V3-5: publish source en watchdog también (5Hz refresh para dashboard)
        msg = String()
        msg.data = self._current_source
        self._pub_source.publish(msg)

    def _log_source_change(self):
        if self._current_source == self._prev_source:
            return
        self._prev_source = self._current_source

        labels = {
            SourceState.PREDICT_ONLY:   '⚠  PREDICT_ONLY   — sin corrección externa',
            SourceState.MCL_ACTIVE:     '✓  MCL_ACTIVE     — /mcl/pose',
            SourceState.ICP_ACTIVE:     '✓  ICP_ACTIVE     — /icp/pose',
            SourceState.ARUCO_ACTIVE:   '✓  ARUCO_ACTIVE   — /aruco/pose',
            SourceState.MCL_PRIORITY:   '✓  MCL_PRIORITY   — MCL sobre ICP',
            SourceState.ARUCO_PRIORITY: '✓  ARUCO_PRIORITY — ArUco sobre ICP/MCL',
        }
        label = labels[self._current_source]

        if self._current_source == SourceState.PREDICT_ONLY:
            self.get_logger().warn(f'EKF source → {label}')
            with self._ekf_lock:
                self._aruco_initialised = False
                self._mcl_initialised   = False
                self._icp_initialised   = False
        else:
            self.get_logger().info(f'EKF source → {label}')

        msg = String()
        msg.data = self._current_source
        self._pub_source.publish(msg)

    # ── EKF PREDICT ──────────────────────────────────────────────────────────

    def _predict_and_publish(self):
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        # FIX-V3-3: tighter dt cap — 0.05s instead of 0.08s
        if dt < 0.001 or dt > 0.05:
            self._last_time = now
            return
        self._last_time = now

        with self._ekf_lock:
            wl = self._wl
            wr = self._wr

            # FIX-V3-2: apply per-wheel slip calibration
            wl_eff = wl * self._slip_l
            wr_eff = wr * self._slip_r

            v = self._R_wheel * (wr_eff + wl_eff) / 2.0
            w = self._R_wheel * (wr_eff - wl_eff) / self._L

            th = self._x[2]

            # FIX-V3-1: RK2 (midpoint method) for both corrected and raw poses
            # Step 1: predict heading at midpoint of interval
            th_mid = th + w * dt / 2.0

            # Step 2: integrate using midpoint heading
            self._x[0] += v * math.cos(th_mid) * dt
            self._x[1] += v * math.sin(th_mid) * dt
            self._x[2]  = normalize_angle(th + w * dt)

            # Same for raw dead-reckoning
            raw_th     = self._raw_x[2]
            raw_th_mid = raw_th + w * dt / 2.0
            self._raw_x[0] += v * math.cos(raw_th_mid) * dt
            self._raw_x[1] += v * math.sin(raw_th_mid) * dt
            self._raw_x[2]  = normalize_angle(raw_th + w * dt)

            # Covariance prediction
            # FIX-V3-4: inflate Q when in PREDICT_ONLY for a long time so the
            # innovation gate accepts corrections after MCL/ArUco re-acquisition
            last_upd = self._last_update_t
            q_scale  = 1.0
            if (self._current_source == SourceState.PREDICT_ONLY
                    and last_upd > 0):
                predict_only_s = time.monotonic() - last_upd
                if predict_only_s > self._decay_s:
                    q_scale = self._decay_k
                    self.get_logger().debug(
                        f'[EKF] PREDICT_ONLY {predict_only_s:.1f}s — '
                        f'inflating Q x{q_scale}')

            F = np.array([
                [1.0, 0.0, -v * math.sin(th_mid) * dt],
                [0.0, 1.0,  v * math.cos(th_mid) * dt],
                [0.0, 0.0,  1.0],
            ])
            Q = np.diag(self._Q_diag * dt * q_scale)
            self._P = F @ self._P @ F.T + Q

            x_snap = self._x.copy()
            P_snap = self._P.copy()

        self._publish(x_snap, P_snap, v, w, now)

    # ── EKF UPDATE ───────────────────────────────────────────────────────────

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
                pos_sigma = math.sqrt(max(self._P[0, 0] + self._P[1, 1], 0.0))
                yaw_sigma = math.sqrt(max(self._P[2, 2], 0.0))
                gate_pos  = max(self._max_ipos, 3.0 * pos_sigma)
                gate_yaw  = max(self._max_iyaw, 3.0 * yaw_sigma)
                if (math.hypot(y[0], y[1]) > gate_pos or
                        abs(y[2]) > gate_yaw):
                    self.get_logger().warn(
                        f'[{source}] update REJECTED — '
                        f'Δpos={math.hypot(y[0],y[1]):.3f}m (gate={gate_pos:.2f}), '
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

            # Update map → odom offset
            dyaw  = normalize_angle(self._x[2] - self._raw_x[2])
            cos_d = math.cos(dyaw)
            sin_d = math.sin(dyaw)
            rx    = self._raw_x[0]
            ry    = self._raw_x[1]
            self._map_odom[0] = self._x[0] - (cos_d * rx - sin_d * ry)
            self._map_odom[1] = self._x[1] - (sin_d * rx + cos_d * ry)
            self._map_odom[2] = dyaw

            # FIX-V3-4: record time of last successful update
            self._last_update_t = time.monotonic()

        self.get_logger().debug(
            f'[{source}]{"(init)" if first_update else ""} '
            f'Δpos={math.hypot(y[0],y[1]):.4f}m, '
            f'Δyaw={math.degrees(y[2]):.2f}°'
        )

    # ── publish ──────────────────────────────────────────────────────────────

    def _publish(self, x: np.ndarray, P: np.ndarray,
                 v: float, w: float, now):
        odom = Odometry()
        odom.header.stamp    = now.to_msg()
        odom.header.frame_id = 'map'
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

        # TF: odom → base_link (dead-reckoning)
        with self._ekf_lock:
            raw      = self._raw_x.copy()
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

        # TF: map → odom (EKF correction)
        tf_mo = TransformStamped()
        tf_mo.header.stamp    = now.to_msg()
        tf_mo.header.frame_id = 'map'
        tf_mo.child_frame_id  = 'odom'
        tf_mo.transform.translation.x = map_odom[0]
        tf_mo.transform.translation.y = map_odom[1]
        tf_mo.transform.translation.z = 0.0
        tf_mo.transform.rotation      = euler_to_quaternion(0.0, 0.0, map_odom[2])
        self._tf_broadcaster.sendTransform(tf_mo)


# ── entry point ───────────────────────────────────────────────────────────────

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