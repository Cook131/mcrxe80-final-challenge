#!/usr/bin/env python3
"""
bug_IBA.py  —  Capa Reflejo de Seguridad v3.0  (Híbrido Reactivo + A*)
=======================================================================
Nodo intermedio entre nav_fsm/GoToGoal y puzzlebotController.

Pipeline:
  A* ──/goal──► GoToGoal ──/cmd_raw──► [bug_IBA] ──/cmd_vel──► Controller
                                            ▲  │
                                       /scan  └──/replan_trigger──► A*

Filosofía del híbrido
─────────────────────
El reflex arc (reactivo puro) sigue siendo el primer respondedor: reacciona
al LiDAR en <50 ms sin esperar a ningún planificador. Cuando el robot entra
a REFLEX_TURN publica un trigger al nodo A*, que recalcula una ruta desde
la posición actual al mismo goal final. Mientras el replan no llega el
robot esquiva reactivamente; en cuanto GoToGoal recibe el nuevo waypoint
el control vuelve a PASS_THROUGH y se retoma la trayectoria planificada.

Esto garantiza:
  1. Responsividad máxima ante obstáculos (reflex <50 ms).
  2. Continuidad de misión: siempre se retoma el goal original.
  3. Sin oscilaciones: el replan solo se dispara una vez por evento
     de obstáculo (cooldown configurable).

Modos en cascada de prioridad fija (sin cambios respecto a v2.1):

  PASS_THROUGH      frente > warn_dist
  PREDICTIVE_BRAKE  warn_dist ≥ frente > emergency_dist
  REFLEX_TURN       frente ≤ emergency_dist
  REFLEX_STOP       frente ≤ stop_dist

Cambios v3.0 sobre v2.1:
  [NEW-1]  Suscripción a /odom para conocer posición actual del robot.
  [NEW-2]  Publicador /replan_trigger (Pose2D) — dispara replan en A*.
  [NEW-3]  Suscripción a /astar/status para saber cuándo el replan terminó.
  [NEW-4]  Cooldown de replan configurable (replan_cooldown_s) para evitar
           spam de replans en obstáculos largos.
  [NEW-5]  Estado REPLANNING: durante el replan el reflex sigue activo pero
           no envía nuevos triggers.
  [FIX-B4] Hold de REFLEX_TURN tiene veto sobre REFLEX_STOP si front >
           stop_d puro, eliminando la oscilación TURN↔STOP por ruido LiDAR.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


PASS_THROUGH     = "PASS"
PREDICTIVE_BRAKE = "BRAKE"
REFLEX_TURN      = "REFLEX_TURN"
REFLEX_STOP      = "REFLEX_STOP"


class BugReflex(Node):
    """Subsumption safety layer con braking predictivo, escape en arco y replan A*."""

    def __init__(self):
        super().__init__('bug_reflex')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('warn_dist',           0.55)
        self.declare_parameter('emergency_dist',      0.22)
        self.declare_parameter('stop_dist',           0.10)
        self.declare_parameter('reflex_v',            0.04)
        self.declare_parameter('reflex_w',            0.65)
        self.declare_parameter('reflex_hold_ms',      350)
        self.declare_parameter('front_half_deg',      30.0)
        self.declare_parameter('side_half_deg',       35.0)
        self.declare_parameter('hysteresis',          0.06)
        self.declare_parameter('lidar_yaw_offset',    0.0)
        # [NEW-4] Tiempo mínimo entre replans consecutivos [s]
        self.declare_parameter('replan_cooldown_s',   2.0)

        self.warn_d       = float(self.get_parameter('warn_dist').value)
        self.emg_d        = float(self.get_parameter('emergency_dist').value)
        self.stop_d       = float(self.get_parameter('stop_dist').value)
        self.ref_v        = float(self.get_parameter('reflex_v').value)
        self.ref_w        = float(self.get_parameter('reflex_w').value)
        self.hold_s       = float(self.get_parameter('reflex_hold_ms').value) / 1000.0
        self.front_h      = math.radians(self.get_parameter('front_half_deg').value)
        self.side_h       = math.radians(self.get_parameter('side_half_deg').value)
        self.hyst         = float(self.get_parameter('hysteresis').value)
        self._lidar_yaw_offset = float(self.get_parameter('lidar_yaw_offset').value)
        self._replan_cooldown  = float(self.get_parameter('replan_cooldown_s').value)

        # ── Estado interno ────────────────────────────────────────────────
        self._mode        = PASS_THROUGH
        self._reflex_ts   = 0.0
        self._last_cmd    = Twist()
        self.scan: LaserScan | None = None

        # [NEW-1] Posición del robot (actualizada por /odom)
        self._robot_x     = 0.0
        self._robot_y     = 0.0

        # [NEW-4] Control de replans
        self._last_replan_ts  = -999.0   # timestamp del último replan enviado
        self._replanning      = False     # True mientras A* no ha confirmado fin

        # ── QoS best-effort para LiDAR ────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(Twist,     '/cmd_raw',      self._cb_cmd,    10)
        self.create_subscription(LaserScan, '/scan',         self._cb_scan,   scan_qos)
        # [NEW-1]
        self.create_subscription(Odometry,  '/odom',         self._cb_odom,   10)
        # [NEW-3] Escucha el estado del A* para saber cuándo terminó el replan
        self.create_subscription(String,    '/astar/status', self._cb_astar_status, 10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_cmd     = self.create_publisher(Twist,  '/cmd_vel',          10)
        self._pub_status  = self.create_publisher(String, '/reflex_status',    10)
        # [NEW-2]
        self._pub_replan  = self.create_publisher(Pose2D, '/replan_trigger',   10)

        self.create_timer(0.05, self._loop)   # 20 Hz

        self.get_logger().info(
            f'[BugReflex v3.0] Lista | '
            f'warn={self.warn_d:.2f}m | emg={self.emg_d:.2f}m | '
            f'stop={self.stop_d:.2f}m | replan_cooldown={self._replan_cooldown:.1f}s')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_cmd(self, msg: Twist):
        self._last_cmd = msg

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg

    def _cb_odom(self, msg: Odometry):
        # [NEW-1] Solo necesitamos xy para el trigger de replan
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _cb_astar_status(self, msg: String):
        # [NEW-3] Cuando A* termina de planificar, libera la bandera
        if msg.data in ('EXECUTING', 'GOAL_REACHED', 'NO_PATH'):
            if self._replanning:
                self.get_logger().info(
                    f'[BugReflex] Replan completado (A* status: {msg.data}) — '
                    f'retomando trayectoria planificada.')
            self._replanning = False

    # ── Loop principal ────────────────────────────────────────────────────

    def _loop(self):
        scan = self.scan
        if scan is None:
            self._publish(self._last_cmd, PASS_THROUGH)
            return

        front = self._sector_min(scan, 0.0,                self.front_h)
        left  = self._sector_min(scan, math.radians( 90),  self.side_h)
        right = self._sector_min(scan, math.radians(-90),  self.side_h)

        now         = time.monotonic()
        hold_active = (now - self._reflex_ts) < self.hold_s

        # ── P1: REFLEX_STOP ───────────────────────────────────────────────
        in_stop  = (self._mode == REFLEX_STOP)
        stop_thr = self.stop_d + (self.hyst if in_stop else 0.0)

        # [FIX-B4] Hold de TURN tiene veto sobre STOP si la distancia
        # es mayor al umbral duro (ruido en el borde stop_d).
        in_turn_hold = hold_active and (self._mode == REFLEX_TURN)
        stop_triggered = (front <= stop_thr or (hold_active and in_stop))
        turn_veto      = in_turn_hold and (front > self.stop_d)

        if stop_triggered and not turn_veto:
            if not in_stop:
                self._reflex_ts = now
            self._publish(Twist(), REFLEX_STOP)
            return

        # ── P2: REFLEX_TURN ───────────────────────────────────────────────
        in_turn  = (self._mode == REFLEX_TURN)
        emg_thr  = self.emg_d + (self.hyst if in_turn else 0.0)

        if front <= emg_thr or (hold_active and in_turn):
            if not in_turn:
                self._reflex_ts = now
                # [NEW-2] Dispara replan solo si no estamos en cooldown
                self._maybe_trigger_replan(now)

            turn_sign = +1.0 if left > right else -1.0   # [BUG-1 fix: > no >=]
            cmd = Twist()
            cmd.linear.x  = self.ref_v
            cmd.angular.z = turn_sign * self.ref_w
            self._publish(cmd, REFLEX_TURN)
            return

        # ── P3: PREDICTIVE_BRAKE ──────────────────────────────────────────
        in_brake   = (self._mode == PREDICTIVE_BRAKE)
        warn_thr   = self.warn_d + (self.hyst if in_brake else 0.0)
        incoming_v = self._last_cmd.linear.x

        if front <= warn_thr and incoming_v > 0.0:
            t = 1.0 - (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
            t = max(0.0, min(1.0, t))
            scale = 1.0 - t

            cmd = Twist()
            cmd.linear.x  = incoming_v * scale
            cmd.angular.z = self._last_cmd.angular.z
            self._publish(cmd, PREDICTIVE_BRAKE)
            return

        # ── P4: PASS_THROUGH ──────────────────────────────────────────────
        if self._mode not in (PASS_THROUGH,):
            self.get_logger().info(
                f'[BugReflex] {self._mode} → PASS | frente libre ({front:.2f}m)')
        self._publish(self._last_cmd, PASS_THROUGH)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _maybe_trigger_replan(self, now: float):
        """
        [NEW-2/4] Publica un trigger de replan si el cooldown lo permite.

        El A* planner escucha /replan_trigger (Pose2D con la posición actual)
        y recalcula la ruta al mismo goal final. El robot sigue esquivando
        reactivamente mientras tanto — el replan es asíncrono.
        """
        if self._replanning:
            return
        if (now - self._last_replan_ts) < self._replan_cooldown:
            return

        msg = Pose2D()
        msg.x = self._robot_x
        msg.y = self._robot_y
        self._pub_replan.publish(msg)

        self._replanning     = True
        self._last_replan_ts = now

        self.get_logger().warn(
            f'[BugReflex] Obstáculo detectado → replan disparado desde '
            f'({self._robot_x:.2f}, {self._robot_y:.2f})')

    def _sector_min(self, scan: LaserScan, center_rad: float, half_rad: float) -> float:
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._lidar_yaw_offset)

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