#!/usr/bin/env python3
"""
bug_reflex.py  —  Capa Reflejo de Seguridad v2
================================================
Nodo intermedio entre nav_fsm y puzzlebotController.
Actúa como "reflex arc" del robot: reacciona a colisiones inminentes
sin esperar a la FSM.

Pipeline:
  nav_fsm → /cmd_raw → [bug_reflex] → /cmd_vel → puzzlebotController
                              ▲
                         /scan (alta frecuencia)

Tres modos en cascada de prioridad fija:

  PASS_THROUGH   frente > emergency_dist
                 → cmd_raw pasa sin cambios

  PREDICTIVE_BRAKE  warn_dist > frente > emergency_dist
                 → escala v lineal hacia 0 (sin tocar angular)
                   da tiempo a la FSM de reaccionar antes del reflejo

  REFLEX_TURN    frente ≤ emergency_dist (y no stop_dist)
                 → curva de escape: pequeña v_lin + w hacia espacio libre
                   NO es rotación pura — el robot sigue moviéndose
                   para no atascarse

  REFLEX_STOP    frente ≤ stop_dist
                 → Twist cero (colisión inminente inevitable)

Diferencias vs v1:
  - Zona PREDICTIVE_BRAKE entre PASS y REFLEX (transición suave)
  - REFLEX_TURN usa arco de escape (v > 0) en vez de rotación pura
  - Dirección de giro elegida por lado con más espacio (no fijo)
  - Hysteresis separada por zona para evitar flicker
"""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


PASS_THROUGH      = "PASS"
PREDICTIVE_BRAKE  = "BRAKE"
REFLEX_TURN       = "REFLEX_TURN"
REFLEX_STOP       = "REFLEX_STOP"


class BugReflex(Node):
    """Subsumption safety layer con braking predictivo y escape en arco."""

    def __init__(self):
        super().__init__('bug_reflex')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('warn_dist',       0.55)   # inicio brake zone [m]
        self.declare_parameter('emergency_dist',  0.22)   # activación REFLEX_TURN [m]
        self.declare_parameter('stop_dist',       0.10)   # activación REFLEX_STOP [m]
        self.declare_parameter('reflex_v',        0.04)   # v lineal en arco de escape [m/s]
        self.declare_parameter('reflex_w',        0.65)   # w angular en escape [rad/s]
        self.declare_parameter('reflex_hold_ms',  350)    # hold mínimo del reflejo [ms]
        self.declare_parameter('front_half_deg',  30.0)   # semisector frontal [deg]
        self.declare_parameter('side_half_deg',   35.0)   # semisector lateral [deg]
        self.declare_parameter('hysteresis',      0.06)   # margen de desactivación [m]
        self.declare_parameter('lidar_yaw_offset', 0.0)

        self.warn_d   = float(self.get_parameter('warn_dist').value)
        self.emg_d    = float(self.get_parameter('emergency_dist').value)
        self.stop_d   = float(self.get_parameter('stop_dist').value)
        self.ref_v    = float(self.get_parameter('reflex_v').value)
        self.ref_w    = float(self.get_parameter('reflex_w').value)
        self.hold_s   = float(self.get_parameter('reflex_hold_ms').value) / 1000.0
        self.front_h         = math.radians(self.get_parameter('front_half_deg').value)
        self._lidar_yaw_offset = self.get_parameter('lidar_yaw_offset').value
        self.side_h   = math.radians(self.get_parameter('side_half_deg').value)
        self.hyst     = float(self.get_parameter('hysteresis').value)

        # ── Estado interno ────────────────────────────────────────────────
        self._mode        = PASS_THROUGH
        self._reflex_ts   = 0.0
        self._last_cmd    = Twist()
        self.scan: LaserScan | None = None

        # ── QoS best-effort para LiDAR ────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        self.create_subscription(Twist,     '/cmd_raw', self._cb_cmd,  10)
        self.create_subscription(LaserScan, '/scan',    self._cb_scan, scan_qos)

        self._pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',      10)
        self._pub_status = self.create_publisher(String, '/reflex_status', 10)

        self.create_timer(0.05, self._loop)   # 20 Hz

        self.get_logger().info(
            f'[BugReflex v2] Lista | '
            f'warn={self.warn_d:.2f}m | '
            f'emg={self.emg_d:.2f}m | '
            f'stop={self.stop_d:.2f}m')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_cmd(self, msg: Twist):
        self._last_cmd = msg

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg

    # ── Loop principal ────────────────────────────────────────────────────

    def _loop(self):
        if self.scan is None:
            self._publish(self._last_cmd, PASS_THROUGH)
            return

        front = self._sector_min(0.0,              self.front_h)
        left  = self._sector_min(math.radians(90), self.side_h)
        right = self._sector_min(math.radians(-90), self.side_h)

        now         = time.monotonic()
        hold_active = (now - self._reflex_ts) < self.hold_s

        # ── P1: REFLEX_STOP (colisión inminente) ──────────────────────────
        if front <= self.stop_d:
            if self._mode != REFLEX_STOP:
                self._reflex_ts = now
            self._publish(Twist(), REFLEX_STOP)
            return

        # ── P2: REFLEX_TURN (escape en arco) ──────────────────────────────
        emg_clear = self.emg_d + self.hyst
        if front <= self.emg_d or (hold_active and self._mode == REFLEX_TURN):
            if self._mode != REFLEX_TURN:
                self._reflex_ts = now

            # Elige dirección hacia el lado con más espacio
            turn_sign = +1.0 if left >= right else -1.0

            cmd = Twist()
            cmd.linear.x  = self.ref_v             # arco, no rotación pura
            cmd.angular.z = turn_sign * self.ref_w
            self._publish(cmd, REFLEX_TURN)
            return

        # ── P3: PREDICTIVE_BRAKE (desacelerar suavemente) ─────────────────
        # Solo actúa si el robot se mueve hacia adelante
        warn_clear = self.warn_d + self.hyst
        incoming_v = self._last_cmd.linear.x
        if front <= self.warn_d and incoming_v > 0.0:
            # Fracción de frenado: 0.0 en warn_dist, 1.0 en emergency_dist
            t = 1.0 - (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
            t = max(0.0, min(1.0, t))
            scale = 1.0 - t   # escala de velocidad: 1.0 → 0.0

            cmd = Twist()
            cmd.linear.x  = incoming_v * scale
            cmd.angular.z = self._last_cmd.angular.z   # angular sin cambio
            self._publish(cmd, PREDICTIVE_BRAKE)
            return

        # ── P4: PASS_THROUGH ──────────────────────────────────────────────
        if self._mode not in (PASS_THROUGH, PREDICTIVE_BRAKE):
            self.get_logger().info(
                f'[BugReflex] Reflejo terminado — frente libre ({front:.2f}m)')
        self._publish(self._last_cmd, PASS_THROUGH)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _sector_min(self, center_rad: float, half_rad: float) -> float:
        mn = float('inf')
        for i, r in enumerate(self.scan.ranges):
            a = self.scan.angle_min + i * self.scan.angle_increment + self._lidar_yaw_offset
            if abs(math.atan2(math.sin(a - center_rad),
                              math.cos(a - center_rad))) <= half_rad:
                if self.scan.range_min < r < self.scan.range_max:
                    mn = min(mn, r)
        return mn

    def _publish(self, cmd: Twist, mode: str):
        if mode != self._mode:
            if mode not in (PASS_THROUGH, PREDICTIVE_BRAKE):
                self.get_logger().warn(
                    f'[BugReflex] {self._mode} → {mode}',
                    throttle_duration_sec=0.4)
            self._mode = mode

        self._pub_cmd.publish(cmd)
        s = String(); s.data = mode
        self._pub_status.publish(s)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BugReflex()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._pub_cmd.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()