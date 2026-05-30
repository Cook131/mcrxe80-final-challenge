#!/usr/bin/env python3
"""
bug_reflex.py  —  Capa Reflejo de Seguridad v2.1
=================================================
Nodo intermedio entre nav_fsm y puzzlebotController.
Actúa como "reflex arc" del robot: reacciona a colisiones inminentes
sin esperar a la FSM.

Pipeline:
  nav_fsm → /cmd_raw → [bug_reflex] → /cmd_vel → puzzlebotController
                              ▲
                         /scan (alta frecuencia)

Tres modos en cascada de prioridad fija:

  PASS_THROUGH      frente > warn_dist
                    → cmd_raw pasa sin cambios

  PREDICTIVE_BRAKE  warn_dist ≥ frente > emergency_dist
                    → escala v lineal hacia 0 (sin tocar angular)
                      da tiempo a la FSM de reaccionar antes del reflejo

  REFLEX_TURN       frente ≤ emergency_dist (y no stop_dist)
                    → curva de escape: pequeña v_lin + w hacia espacio libre
                      NO es rotación pura — el robot sigue moviéndose
                      para no atascarse

  REFLEX_STOP       frente ≤ stop_dist
                    → Twist cero (colisión inminente inevitable)

Fixes v2.1 sobre v2:
  [FIX-1] _sector_min vectorizado con NumPy (O(n) → O(1) efectivo)
  [FIX-2] Histéresis de REFLEX_TURN y PREDICTIVE_BRAKE realmente aplicada
          (emg_clear / warn_clear antes eran calculados pero nunca usados)
  [FIX-3] REFLEX_STOP ahora tiene hold + histéresis igual que REFLEX_TURN
  [FIX-4] self.scan leído con snapshot local en _loop para seguridad en hilos
  [FIX-5] Transición BRAKE → PASS logeada correctamente
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


PASS_THROUGH     = "PASS"
PREDICTIVE_BRAKE = "BRAKE"
REFLEX_TURN      = "REFLEX_TURN"
REFLEX_STOP      = "REFLEX_STOP"


class BugReflex(Node):
    """Subsumption safety layer con braking predictivo y escape en arco."""

    def __init__(self):
        super().__init__('bug_reflex')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('warn_dist',        0.55)   # inicio brake zone [m]
        self.declare_parameter('emergency_dist',   0.22)   # activación REFLEX_TURN [m]
        self.declare_parameter('stop_dist',        0.10)   # activación REFLEX_STOP [m]
        self.declare_parameter('reflex_v',         0.04)   # v lineal en arco de escape [m/s]
        self.declare_parameter('reflex_w',         0.65)   # w angular en escape [rad/s]
        self.declare_parameter('reflex_hold_ms',   350)    # hold mínimo del reflejo [ms]
        self.declare_parameter('front_half_deg',   30.0)   # semisector frontal [deg]
        self.declare_parameter('side_half_deg',    35.0)   # semisector lateral [deg]
        self.declare_parameter('hysteresis',       0.06)   # margen de desactivación [m]
        self.declare_parameter('lidar_yaw_offset', 0.0)

        self.warn_d  = float(self.get_parameter('warn_dist').value)
        self.emg_d   = float(self.get_parameter('emergency_dist').value)
        self.stop_d  = float(self.get_parameter('stop_dist').value)
        self.ref_v   = float(self.get_parameter('reflex_v').value)
        self.ref_w   = float(self.get_parameter('reflex_w').value)
        self.hold_s  = float(self.get_parameter('reflex_hold_ms').value) / 1000.0
        self.front_h = math.radians(self.get_parameter('front_half_deg').value)
        self.side_h  = math.radians(self.get_parameter('side_half_deg').value)
        self.hyst    = float(self.get_parameter('hysteresis').value)
        self._lidar_yaw_offset = float(self.get_parameter('lidar_yaw_offset').value)

        # ── Estado interno ────────────────────────────────────────────────
        self._mode      = PASS_THROUGH
        self._reflex_ts = 0.0
        self._last_cmd  = Twist()
        self.scan: LaserScan | None = None

        # ── QoS best-effort para LiDAR ────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        self.create_subscription(Twist,     '/cmd_raw', self._cb_cmd,  10)
        self.create_subscription(LaserScan, '/scan',    self._cb_scan, scan_qos)

        self._pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',       10)
        self._pub_status = self.create_publisher(String, '/reflex_status',  10)

        self.create_timer(0.05, self._loop)   # 20 Hz

        self.get_logger().info(
            f'[BugReflex v2.1] Lista | '
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
        # [FIX-4] Snapshot local del scan para evitar condición de carrera
        # entre el callback del executor y el timer.
        scan = self.scan
        if scan is None:
            self._publish(self._last_cmd, PASS_THROUGH)
            return

        front = self._sector_min(scan, 0.0,               self.front_h)
        left  = self._sector_min(scan, math.radians( 90), self.side_h)
        right = self._sector_min(scan, math.radians(-90), self.side_h)

        now         = time.monotonic()
        hold_active = (now - self._reflex_ts) < self.hold_s

        # ── P1: REFLEX_STOP ───────────────────────────────────────────────
        # [FIX-3] Ahora tiene hold + histéresis igual que REFLEX_TURN.
        # Sin esto, una lectura de LiDAR ruidosa causa stop/resume errático.
        in_stop  = (self._mode == REFLEX_STOP)
        stop_thr = self.stop_d + (self.hyst if in_stop else 0.0)

        if front <= stop_thr or (hold_active and in_stop):
            if not in_stop:
                self._reflex_ts = now
            self._publish(Twist(), REFLEX_STOP)
            return

        # ── P2: REFLEX_TURN ───────────────────────────────────────────────
        # [FIX-2] emg_clear ahora se usa realmente como umbral de salida.
        # Antes se calculaba pero la condición seguía usando self.emg_d puro,
        # dejando la histéresis completamente inoperativa.
        in_turn  = (self._mode == REFLEX_TURN)
        emg_thr  = self.emg_d + (self.hyst if in_turn else 0.0)

        if front <= emg_thr or (hold_active and in_turn):
            if not in_turn:
                self._reflex_ts = now

            turn_sign = +1.0 if left >= right else -1.0
            cmd = Twist()
            cmd.linear.x  = self.ref_v
            cmd.angular.z = turn_sign * self.ref_w
            self._publish(cmd, REFLEX_TURN)
            return

        # ── P3: PREDICTIVE_BRAKE ──────────────────────────────────────────
        # [FIX-2] warn_clear ahora se usa realmente como umbral de salida.
        # Solo actúa si el robot se mueve hacia adelante.
        in_brake  = (self._mode == PREDICTIVE_BRAKE)
        warn_thr  = self.warn_d + (self.hyst if in_brake else 0.0)
        incoming_v = self._last_cmd.linear.x

        if front <= warn_thr and incoming_v > 0.0:
            # Fracción: 0.0 en warn_dist, 1.0 en emergency_dist
            t = 1.0 - (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
            t = max(0.0, min(1.0, t))
            scale = 1.0 - t   # escala de velocidad: 1.0 → 0.0

            cmd = Twist()
            cmd.linear.x  = incoming_v * scale
            cmd.angular.z = self._last_cmd.angular.z
            self._publish(cmd, PREDICTIVE_BRAKE)
            return

        # ── P4: PASS_THROUGH ──────────────────────────────────────────────
        # [FIX-5] Log de salida ahora cubre también la transición BRAKE→PASS,
        # no solo los modos de reflejo duro.
        if self._mode not in (PASS_THROUGH,):
            self.get_logger().info(
                f'[BugReflex] {self._mode} → PASS | frente libre ({front:.2f}m)')
        self._publish(self._last_cmd, PASS_THROUGH)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _sector_min(self, scan: LaserScan, center_rad: float, half_rad: float) -> float:
        """
        [FIX-1] Versión vectorizada con NumPy.

        La versión original iteraba con un for-loop de Python sobre todos los
        rangos en cada llamada. Con un RPLIDAR A1 (~8000 muestras a 10 Hz)
        llamado 3 veces por ciclo a 20 Hz, eso equivale a ~480 000 iteraciones
        de Python por segundo — un cuello de botella real en el Jetson Nano.

        Esta versión construye los arrays una sola vez por ciclo de scan,
        aplica la máscara de sector y retorna el mínimo en operaciones NumPy.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._lidar_yaw_offset)

        # Diferencia angular envuelta a [-π, π]
        diff = np.arctan2(np.sin(angles - center_rad),
                          np.cos(angles - center_rad))

        mask = (
            (np.abs(diff) <= half_rad) &
            (ranges > scan.range_min) &
            (ranges < scan.range_max)
        )

        valid = ranges[mask]
        return float(np.min(valid)) if valid.size > 0 else float('inf')

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