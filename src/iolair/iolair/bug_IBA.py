#!/usr/bin/env python3
"""
bug_IBA.py  —  Capa Reflejo de Seguridad v4.0  (BUG2 + A* Replan)
==================================================================
Nodo intermedio entre nav_fsm/GoToGoal y puzzlebotController.

Pipeline:
  A* ──/goal──► GoToGoal ──/cmd_raw──► [bug_IBA] ──/cmd_vel──► Controller
                                            ▲  │
                                       /scan  └──/replan_trigger──► A*

Filosofía del híbrido BUG2 + A*
────────────────────────────────
1. En condiciones normales el robot sigue la ruta A* en PASS_THROUGH.
2. Al detectar obstáculo (<= emergency_dist) entra a BUG2_WALL_FOLLOW:
   - Se registra el hit_point y la distancia hit→goal (línea M de BUG2).
   - El robot bordea el contorno del obstáculo (wall-follow reactivo).
   - Simultáneamente se dispara un replan asíncrono al A*.
3. Condición de salida BUG2 (clásica):
   - El robot cruza la línea M (distancia perpendicular < m_line_tol).
   - Está más cerca al goal que en el hit_point (progreso real).
   - El frente está libre (>= warn_dist) — no re-choca al salir.
4. Al salir de BUG2_WALL_FOLLOW vuelve a PASS_THROUGH; si el A* ya
   replanificó, GoToGoal sigue los nuevos waypoints automáticamente.

Modos (cascada de prioridad fija):

  PASS_THROUGH       frente > warn_dist  (o BUG2 salida)
  PREDICTIVE_BRAKE   warn_dist ≥ frente > emergency_dist
  BUG2_WALL_FOLLOW   frente ≤ emergency_dist  (bordeo de obstáculo BUG2)
  REFLEX_STOP        frente ≤ stop_dist  (parada de emergencia dura)

Cambios v4.0 sobre v3.0:
  [BUG2-1]  REFLEX_TURN reemplazado por BUG2_WALL_FOLLOW.
  [BUG2-2]  Suscripción a /astar/goal para conocer el goal final y
            calcular la línea M en el momento del hit.
  [BUG2-3]  _check_bug2_exit(): condición clásica de salida BUG2.
  [BUG2-4]  Memoria de dirección de giro (_last_turn_sign) para
            desempate en corredores simétricos.
  [FIX-v3]  stop_triggered sin hold (corregido en v3.0 fix).
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
BUG2_WALL_FOLLOW = "BUG2_WALL"
REFLEX_STOP      = "REFLEX_STOP"


class BugReflex(Node):
    """Subsumption safety layer con BUG2, braking predictivo y replan A*."""

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
        self.declare_parameter('lidar_yaw_offset',    math.pi)  # LiDAR montado invertido — igual que SLAM/MCL
        self.declare_parameter('replan_cooldown_s',   2.0)
        # [BUG2-3] Tolerancia para considerar que el robot está sobre la línea M [m]
        self.declare_parameter('m_line_tol',          0.12)
        # [BUG2-3] Distancia mínima recorrida en wall-follow antes de chequear salida
        self.declare_parameter('bug2_min_follow_m',   0.20)

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
        self._m_line_tol       = float(self.get_parameter('m_line_tol').value)
        self._bug2_min_follow  = float(self.get_parameter('bug2_min_follow_m').value)

        # ── Estado interno ────────────────────────────────────────────────
        self._mode        = PASS_THROUGH
        self._reflex_ts   = 0.0
        self._last_cmd    = Twist()
        self.scan: LaserScan | None = None

        # Posición del robot (actualizada por /odom)
        self._robot_x     = 0.0
        self._robot_y     = 0.0

        # Control de replans
        self._last_replan_ts  = -999.0
        self._replanning      = False

        # Memoria de dirección de giro (desempate en corredor simétrico)
        self._last_turn_sign  = +1.0

        # [BUG2-2] Goal final para calcular la línea M
        self._goal_x: float | None = None
        self._goal_y: float | None = None

        # [BUG2-1] Estado BUG2: hit point y distancia al goal en el momento del hit
        self._hit_x          = 0.0
        self._hit_y          = 0.0
        self._hit_dist_goal  = float('inf')   # dist(hit_point, goal)
        self._bug2_traveled  = 0.0            # distancia recorrida en wall-follow
        self._bug2_prev_x    = 0.0
        self._bug2_prev_y    = 0.0

        # ── QoS best-effort para LiDAR ────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        # ── Suscriptores ──────────────────────────────────────────────────
        self.create_subscription(Twist,     '/cmd_raw',      self._cb_cmd,         10)
        self.create_subscription(LaserScan, '/scan',         self._cb_scan,        scan_qos)
        self.create_subscription(Odometry,  '/odom',         self._cb_odom,        10)
        self.create_subscription(String,    '/astar/status', self._cb_astar_status, 10)
        # [BUG2-2] Goal final para línea M
        self.create_subscription(Pose2D,    '/astar/goal',   self._cb_goal,        10)

        # ── Publicadores ──────────────────────────────────────────────────
        self._pub_cmd     = self.create_publisher(Twist,  '/cmd_vel',        10)
        self._pub_status  = self.create_publisher(String, '/reflex_status',  10)
        self._pub_replan  = self.create_publisher(Pose2D, '/replan_trigger', 10)

        self.create_timer(0.05, self._loop)   # 20 Hz

        self.get_logger().info(
            f'[BugReflex v4.0 — BUG2] Lista | '
            f'warn={self.warn_d:.2f}m | emg={self.emg_d:.2f}m | '
            f'stop={self.stop_d:.2f}m | m_line_tol={self._m_line_tol:.2f}m')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_cmd(self, msg: Twist):
        self._last_cmd = msg

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg

    def _cb_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _cb_goal(self, msg: Pose2D):
        # [BUG2-2] Actualiza el goal final; resetea estado BUG2 si cambia el goal
        if msg.x != self._goal_x or msg.y != self._goal_y:
            self._goal_x = msg.x
            self._goal_y = msg.y
            self.get_logger().info(
                f'[BugReflex] Goal final actualizado: ({msg.x:.2f}, {msg.y:.2f})')

    def _cb_astar_status(self, msg: String):
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

        front = self._sector_min(scan, 0.0,               self.front_h)
        left  = self._sector_min(scan, math.radians( 90), self.side_h)
        right = self._sector_min(scan, math.radians(-90), self.side_h)

        now         = time.monotonic()
        hold_active = (now - self._reflex_ts) < self.hold_s

        # ── P1: REFLEX_STOP ───────────────────────────────────────────────
        in_stop  = (self._mode == REFLEX_STOP)
        stop_thr = self.stop_d + (self.hyst if in_stop else 0.0)

        in_wall_hold = hold_active and (self._mode == BUG2_WALL_FOLLOW)
        stop_triggered = (front <= stop_thr)               # [FIX-v3] sin hold
        turn_veto      = in_wall_hold and (front > self.stop_d)

        if stop_triggered and not turn_veto:
            if not in_stop:
                self._reflex_ts = now
            self._publish(Twist(), REFLEX_STOP)
            return

        # ── P2: BUG2_WALL_FOLLOW ──────────────────────────────────────────
        in_wall  = (self._mode == BUG2_WALL_FOLLOW)
        emg_thr  = self.emg_d + (self.hyst if in_wall else 0.0)

        if front <= emg_thr or (hold_active and in_wall):
            if not in_wall:
                # Entrada al wall-follow: registrar hit point
                self._reflex_ts      = now
                self._hit_x          = self._robot_x
                self._hit_y          = self._robot_y
                self._hit_dist_goal  = self._dist_to_goal(self._robot_x, self._robot_y)
                self._bug2_traveled  = 0.0
                self._bug2_prev_x    = self._robot_x
                self._bug2_prev_y    = self._robot_y
                self._maybe_trigger_replan(now)
                self.get_logger().warn(
                    f'[BugReflex] BUG2 hit en ({self._hit_x:.2f}, {self._hit_y:.2f}) | '
                    f'dist_goal={self._hit_dist_goal:.2f}m')
            else:
                # Acumular distancia recorrida en wall-follow
                step = math.hypot(self._robot_x - self._bug2_prev_x,
                                  self._robot_y - self._bug2_prev_y)
                self._bug2_traveled += step
                self._bug2_prev_x    = self._robot_x
                self._bug2_prev_y    = self._robot_y

                # [BUG2-3] Chequear condición de salida BUG2
                if self._check_bug2_exit(front):
                    self.get_logger().info(
                        f'[BugReflex] BUG2 salida — sobre línea M, '
                        f'dist_goal={self._dist_to_goal(self._robot_x, self._robot_y):.2f}m '
                        f'< hit={self._hit_dist_goal:.2f}m')
                    self._publish(self._last_cmd, PASS_THROUGH)
                    return

            # Wall-follow: bordear con dirección hacia el lado más libre
            if left != right:
                self._last_turn_sign = +1.0 if left > right else -1.0
            turn_sign = self._last_turn_sign

            cmd = Twist()
            cmd.linear.x  = self.ref_v
            cmd.angular.z = turn_sign * self.ref_w
            self._publish(cmd, BUG2_WALL_FOLLOW)
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

    # ── BUG2: condición de salida ─────────────────────────────────────────

    def _check_bug2_exit(self, front: float) -> bool:
        """
        [BUG2-3] Condición clásica de salida BUG2:
          1. El robot ha recorrido suficiente distancia (evitar salir en el hit point).
          2. Está sobre la línea M (distancia perpendicular < m_line_tol).
          3. Está más cerca al goal que cuando entró al wall-follow.
          4. El frente está libre (no re-choca al retomar rumbo al goal).
        """
        if self._goal_x is None:
            return False

        # 1. Distancia mínima recorrida
        if self._bug2_traveled < self._bug2_min_follow:
            return False

        # 2. Distancia perpendicular a la línea M (hit_point → goal)
        if not self._on_m_line(self._robot_x, self._robot_y):
            return False

        # 3. Más cerca al goal que en el hit point
        curr_dist = self._dist_to_goal(self._robot_x, self._robot_y)
        if curr_dist >= self._hit_dist_goal:
            return False

        # 4. Frente libre para retomar navegación
        if front < self.warn_d:
            return False

        return True

    def _on_m_line(self, px: float, py: float) -> bool:
        """
        Distancia perpendicular del punto (px, py) a la línea
        definida por (hit_x, hit_y) → (goal_x, goal_y).
        Devuelve True si está dentro de m_line_tol.
        """
        if self._goal_x is None:
            return False

        ax, ay = self._hit_x, self._hit_y
        bx, by = self._goal_x, self._goal_y

        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)

        if seg_len < 1e-6:
            # hit_point ≈ goal: cualquier posición se considera sobre la línea
            return True

        # Distancia perpendicular por producto vectorial
        cross = abs(dx * (ay - py) - dy * (ax - px))
        dist_perp = cross / seg_len

        # Adicionalmente chequear que la proyección caiga ENTRE hit y goal
        # (no en la extensión detrás del hit point)
        t = ((px - ax) * dx + (py - ay) * dy) / (seg_len * seg_len)

        return dist_perp < self._m_line_tol and t > 0.0

    def _dist_to_goal(self, x: float, y: float) -> float:
        if self._goal_x is None:
            return float('inf')
        return math.hypot(self._goal_x - x, self._goal_y - y)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _maybe_trigger_replan(self, now: float):
        """
        Publica trigger de replan si el cooldown lo permite.
        El A* replanifica desde la posición actual al mismo goal final.
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
            f'[BugReflex] Replan disparado desde '
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