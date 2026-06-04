#!/usr/bin/env python3
"""
bug_tangent.py  —  Capa Reflejo de Seguridad v5.1  (Tangent Bug + A* Replan)
=============================================================================
Nodo intermedio entre nav_fsm/GoToGoal y puzzlebotController.

Pipeline (drop-in replacement para bug_IBA.py):
  A* ──/goal──► GoToGoal ──/cmd_raw──► [bug_tangent] ──/cmd_vel──► Controller
                                              ▲  │
                                         /scan  └──/replan_trigger──► A*

Algoritmo: Tangent Bug
──────────────────────
Detecta discontinuidades en el LiDAR ("puntos tangentes") que representan los
bordes de los obstáculos, y evalúa la heurística d() en cada uno:

    d(robot→tangente) + d(tangente→goal) < d(robot→goal) * (1 - heuristic_margin)

Cuando existe un punto tangente que satisface esta condición Y el gap hacia
él es físicamente transitable (ancho ≥ 2 * robot_radius * safety_factor),
el robot sale del wall-follow inmediatamente y dispara un replan A*.

Cambios v5.1 sobre v5.0 — "Robot-aware gap filtering"
──────────────────────────────────────────────────────
[FIX-R1]  RADIO DEL ROBOT EXPLÍCITO (robot_radius_m):
          Todos los umbrales de seguridad se derivan de este parámetro.
          gap_min_width_m   → reemplazado por 2 * robot_radius * gap_safety_factor
          wall_follow_dist  → forzado a ≥ robot_radius + wall_clearance_m
          emergency_dist    → forzado a ≥ robot_radius (el cuerpo no puede penetrar)

[FIX-R2]  ESTIMACIÓN DE ANCHO DE GAP CORREGIDA:
          Se mide el ancho transitable real del gap proyectando la apertura
          angular al rango del punto de entrada, en lugar de la cuerda entre
          los dos rayos borde. Esto es más preciso a distancias variables.
          Fórmula: gap_width = r * 2 * sin(delta_angle / 2)
          donde r es la distancia al borde cercano del gap y delta_angle
          es el arco angular entre los dos rayos que lo delimitan.

[FIX-R3]  CLEARANCE LATERAL EN WALL-FOLLOW:
          El P-controller lateral usa wall_follow_dist, pero ahora tiene un
          floor explícito de robot_radius + wall_clearance_m para que nunca
          el borde del robot roce la pared aunque el parámetro sea mal tuneado.

Estados (por prioridad, más alto primero):
  P1  REFLEX_STOP      front ≤ stop_dist → publica Twist() cero
  P2  TANGENT_WALL     front ≤ emg_dist  → wall-follow con P-controller lateral
                       salida por heurística d() sobre gap tangente
  P3  PREDICTIVE_BRAKE front ≤ warn_dist → frena proporcionalmente
  P4  PASS_THROUGH     pasa /cmd_raw sin modificar
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

# ── Constantes de estado ──────────────────────────────────────────────────────
PASS_THROUGH     = "PASS"
PREDICTIVE_BRAKE = "BRAKE"
TANGENT_WALL     = "TANGENT_WALL"
REFLEX_STOP      = "REFLEX_STOP"


@dataclass
class TangentGap:
    """Representa un gap navegable detectado en el LiDAR."""
    bearing: float        # Ángulo del borde del gap [rad, frame robot]
    gap_x: float          # Posición estimada del borde en frame mundo [m]
    gap_y: float          # Posición estimada del borde en frame mundo [m]
    d_heuristic: float    # d(robot→gap) + d(gap→goal)
    width_m: float        # Ancho transitable estimado del gap [m]


class TangentBugReflex(Node):
    """
    Subsumption safety layer con Tangent Bug, braking predictivo y replan A*.
    Drop-in replacement para BugReflex (bug_IBA.py v4.1).
    """

    def __init__(self):
        super().__init__('bug_tangent')

        # ── Parámetros: geometría del robot ───────────────────────────────────
        # Radio de circunscripción del Puzzlebot (mitad de la diagonal) [m].
        # Mídelo físicamente: sqrt((largo/2)^2 + (ancho/2)^2).
        # Todos los umbrales de clearance se derivan de este valor.
        self.declare_parameter('robot_radius_m',      0.18)

        # Factor de seguridad multiplicado por 2*robot_radius para exigir
        # que el gap sea más ancho que el robot. 1.5 = 50% de margen extra.
        self.declare_parameter('gap_safety_factor',   1.5)

        # Clearance lateral mínimo entre el borde del robot y la pared [m].
        # wall_follow_dist se fuerza a ≥ robot_radius + wall_clearance_m.
        self.declare_parameter('wall_clearance_m',    0.08)

        # ── Parámetros heredados de v4.1 (nombres compatibles) ────────────────
        self.declare_parameter('warn_dist',           0.65)
        self.declare_parameter('emergency_dist',      0.35)
        self.declare_parameter('stop_dist',           0.14)
        self.declare_parameter('reflex_v',            0.06)
        self.declare_parameter('reflex_w',            0.65)
        self.declare_parameter('reflex_hold_ms',      350)
        self.declare_parameter('front_half_deg',      30.0)
        self.declare_parameter('side_half_deg',       35.0)
        self.declare_parameter('hysteresis',          0.06)
        self.declare_parameter('lidar_yaw_offset',    math.pi)
        self.declare_parameter('replan_cooldown_s',   2.0)
        self.declare_parameter('wall_follow_dist',    0.40)
        self.declare_parameter('wall_follow_kp',      1.20)
        self.declare_parameter('wall_follow_w_max',   0.80)

        # ── Parámetros Tangent Bug ─────────────────────────────────────────────
        # Relación mínima r[j]/r[i] entre rayos adyacentes para detectar un gap.
        # 1.3 → el rayo libre debe ser ≥30% más largo que el rayo del obstáculo.
        self.declare_parameter('gap_jump_ratio',      1.30)

        # Margen de la heurística: el gap debe ahorrar al menos este porcentaje
        # sobre la distancia directa al goal. 0.10 = 10% de ahorro mínimo.
        self.declare_parameter('heuristic_margin',    0.10)

        # Semisector de búsqueda de gaps a izquierda y derecha [deg].
        self.declare_parameter('tangent_sector_deg',  120.0)

        # Distancia mínima de wall-follow antes de evaluar gaps de salida [m].
        self.declare_parameter('min_follow_m',        0.25)

        # ── Leer y validar parámetros ──────────────────────────────────────────
        self._robot_r      = float(self.get_parameter('robot_radius_m').value)
        self._gap_safety   = float(self.get_parameter('gap_safety_factor').value)
        self._wall_clr     = float(self.get_parameter('wall_clearance_m').value)

        # [FIX-R1] gap_min_width_m derivado del radio del robot
        self._gap_min_w    = 2.0 * self._robot_r * self._gap_safety

        self.warn_d        = float(self.get_parameter('warn_dist').value)
        # [FIX-R1] emergency_dist nunca puede ser menor que el radio del robot
        self.emg_d         = max(
            float(self.get_parameter('emergency_dist').value),
            self._robot_r
        )
        # [FIX-R1] stop_dist nunca puede ser negativo (sanity check)
        self.stop_d        = max(float(self.get_parameter('stop_dist').value), 0.05)

        self.ref_v         = float(self.get_parameter('reflex_v').value)
        self.ref_w         = float(self.get_parameter('reflex_w').value)
        self.hold_s        = float(self.get_parameter('reflex_hold_ms').value) / 1000.0
        self.front_h       = math.radians(self.get_parameter('front_half_deg').value)
        self.side_h        = math.radians(self.get_parameter('side_half_deg').value)
        self.hyst          = float(self.get_parameter('hysteresis').value)
        self._lidar_yaw    = float(self.get_parameter('lidar_yaw_offset').value)
        self._replan_cd    = float(self.get_parameter('replan_cooldown_s').value)

        # [FIX-R3] wall_follow_dist forzado a ≥ robot_radius + wall_clearance
        self._wf_dist      = max(
            float(self.get_parameter('wall_follow_dist').value),
            self._robot_r + self._wall_clr
        )
        self._wf_kp        = float(self.get_parameter('wall_follow_kp').value)
        self._wf_w_max     = float(self.get_parameter('wall_follow_w_max').value)

        self._gap_ratio    = float(self.get_parameter('gap_jump_ratio').value)
        self._h_margin     = float(self.get_parameter('heuristic_margin').value)
        self._tang_sector  = math.radians(self.get_parameter('tangent_sector_deg').value)
        self._min_follow   = float(self.get_parameter('min_follow_m').value)

        # ── Estado interno ────────────────────────────────────────────────────
        self._mode           = PASS_THROUGH
        self._reflex_ts      = 0.0
        self._last_cmd       = Twist()
        self.scan: Optional[LaserScan] = None

        self._robot_x        = 0.0
        self._robot_y        = 0.0
        self._robot_yaw      = 0.0

        self._last_replan_ts = -999.0
        self._replanning     = False
        self._last_turn_sign = +1.0

        self._goal_x: Optional[float] = None
        self._goal_y: Optional[float] = None

        # Estado wall-follow
        self._hit_x          = 0.0
        self._hit_y          = 0.0
        self._traveled       = 0.0
        self._prev_x         = 0.0
        self._prev_y         = 0.0
        self._turn_sign      = +1.0
        self._best_gap: Optional[TangentGap] = None

        # ── QoS best-effort para LiDAR ────────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        # ── Suscriptores ──────────────────────────────────────────────────────
        self.create_subscription(Twist,     '/cmd_raw',      self._cb_cmd,   10)
        self.create_subscription(LaserScan, '/scan',         self._cb_scan,  scan_qos)
        self.create_subscription(Odometry,  '/odom',         self._cb_odom,  10)
        self.create_subscription(String,    '/astar/status', self._cb_astar, 10)
        self.create_subscription(Pose2D,    '/astar/goal',   self._cb_goal,  10)

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_cmd       = self.create_publisher(Twist,  '/cmd_vel',        10)
        self._pub_status    = self.create_publisher(String, '/reflex_status',  10)
        self._pub_replan    = self.create_publisher(Pose2D, '/replan_trigger', 10)
        self._pub_nav_pause = self.create_publisher(Bool,   '/nav_pause',      10)

        self.create_timer(0.05, self._loop)   # 20 Hz

        # Log efectivo para confirmar los floors aplicados
        self.get_logger().info(
            f'[TangentBug v5.1] Lista\n'
            f'  robot_radius    = {self._robot_r:.3f} m\n'
            f'  gap_min_width   = {self._gap_min_w:.3f} m  '
            f'(2 * {self._robot_r:.3f} * {self._gap_safety:.1f})\n'
            f'  wall_follow_dist= {self._wf_dist:.3f} m  '
            f'(floor = robot_r + wall_clr = {self._robot_r + self._wall_clr:.3f})\n'
            f'  emergency_dist  = {self.emg_d:.3f} m  '
            f'(floor = robot_r = {self._robot_r:.3f})\n'
            f'  warn_dist       = {self.warn_d:.3f} m\n'
            f'  stop_dist       = {self.stop_d:.3f} m'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_cmd(self, msg: Twist):
        self._last_cmd = msg

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg

    def _cb_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _cb_goal(self, msg: Pose2D):
        if msg.x != self._goal_x or msg.y != self._goal_y:
            self._goal_x = msg.x
            self._goal_y = msg.y
            self.get_logger().info(
                f'[TangentBug] Goal actualizado: ({msg.x:.2f}, {msg.y:.2f})')

    def _cb_astar(self, msg: String):
        if msg.data in ('EXECUTING', 'GOAL_REACHED', 'NO_PATH'):
            if self._replanning:
                self.get_logger().info(
                    f'[TangentBug] Replan completado (A*: {msg.data})')
            self._replanning = False

    # ── Loop principal ────────────────────────────────────────────────────────

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

        # ── P1: REFLEX_STOP ───────────────────────────────────────────────────
        in_stop  = (self._mode == REFLEX_STOP)
        stop_thr = self.stop_d + (self.hyst if in_stop else 0.0)

        in_wall_hold   = hold_active and (self._mode == TANGENT_WALL)
        stop_triggered = (front <= stop_thr)
        turn_veto      = in_wall_hold and (front > self.stop_d)

        if stop_triggered and not turn_veto:
            if not in_stop:
                self._reflex_ts = now
            self._publish(Twist(), REFLEX_STOP)
            return

        # ── P2: TANGENT_WALL ──────────────────────────────────────────────────
        in_wall = (self._mode == TANGENT_WALL)
        emg_thr = self.emg_d + (self.hyst if in_wall else 0.0)

        if front <= emg_thr or (hold_active and in_wall):
            if not in_wall:
                self._enter_wall_follow(left, right, now)
            else:
                step = math.hypot(self._robot_x - self._prev_x,
                                  self._robot_y - self._prev_y)
                self._traveled += step
                self._prev_x    = self._robot_x
                self._prev_y    = self._robot_y

                if self._traveled >= self._min_follow:
                    gap = self._best_tangent_gap(scan)
                    if gap is not None:
                        self.get_logger().info(
                            f'[TangentBug] Salida por gap tangente | '
                            f'bearing={math.degrees(gap.bearing):.1f}° | '
                            f'd_heur={gap.d_heuristic:.2f}m vs '
                            f'd_direct={self._dist_to_goal(self._robot_x, self._robot_y):.2f}m | '
                            f'ancho={gap.width_m:.2f}m (min={self._gap_min_w:.2f}m)')
                        self._best_gap = gap
                        self._maybe_trigger_replan(now)
                        self._pub_nav_pause.publish(self._bool_msg(False))
                        self._publish(self._last_cmd, PASS_THROUGH)
                        return

            cmd = self._wall_follow_cmd(front, left, right)
            self._publish(cmd, TANGENT_WALL)
            return

        # ── P3: PREDICTIVE_BRAKE ──────────────────────────────────────────────
        in_brake   = (self._mode == PREDICTIVE_BRAKE)
        warn_thr   = self.warn_d + (self.hyst if in_brake else 0.0)
        incoming_v = self._last_cmd.linear.x

        if front <= warn_thr and incoming_v > 0.0:
            t   = 1.0 - (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
            t   = max(0.0, min(1.0, t))
            cmd = Twist()
            cmd.linear.x  = incoming_v * (1.0 - t)
            cmd.angular.z = self._last_cmd.angular.z
            self._publish(cmd, PREDICTIVE_BRAKE)
            return

        # ── P4: PASS_THROUGH ──────────────────────────────────────────────────
        if self._mode not in (PASS_THROUGH,):
            self.get_logger().info(
                f'[TangentBug] {self._mode} → PASS | frente={front:.2f}m')
        self._publish(self._last_cmd, PASS_THROUGH)

    # ── Entrada al wall-follow ────────────────────────────────────────────────

    def _enter_wall_follow(self, left: float, right: float, now: float):
        self._reflex_ts = now
        self._hit_x     = self._robot_x
        self._hit_y     = self._robot_y
        self._traveled  = 0.0
        self._prev_x    = self._robot_x
        self._prev_y    = self._robot_y
        self._best_gap  = None

        if abs(left - right) > 0.05:
            self._turn_sign = +1.0 if left > right else -1.0
        else:
            self._turn_sign = self._last_turn_sign
        self._last_turn_sign = self._turn_sign

        self._pub_nav_pause.publish(self._bool_msg(True))
        self.get_logger().warn(
            f'[TangentBug] Hit en ({self._hit_x:.2f}, {self._hit_y:.2f}) | '
            f'bordeo={"IZQ" if self._turn_sign > 0 else "DER"} — nav PAUSADA')

    # ── Heurística Tangent Bug: detección de gaps robot-aware ────────────────

    def _best_tangent_gap(self, scan: LaserScan) -> Optional[TangentGap]:
        """
        Detecta discontinuidades en el LiDAR y devuelve el gap con mejor
        heurística d() que sea físicamente transitable por el robot.

        [FIX-R2] Ancho del gap:
          Se calcula como la cuerda del arco angular subtendido por el gap
          a la distancia del rayo más corto (borde del obstáculo):

              gap_width = r_near * 2 * sin(delta_angle / 2)

          donde delta_angle es el ángulo entre los dos rayos que delimitan
          el gap. Esto mide el ancho real de la apertura en el plano del
          obstáculo, no la distancia entre puntos en el espacio 3D.

        [FIX-R1] Umbral de ancho:
          gap_width ≥ 2 * robot_radius * gap_safety_factor
        """
        if self._goal_x is None:
            return None

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._lidar_yaw)
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        sector_mask = (
            (np.abs(angles) <= self._tang_sector) &
            (ranges > scan.range_min) &
            (ranges < scan.range_max)
        )
        valid_idx = np.where(sector_mask)[0]
        if valid_idx.size < 2:
            return None

        d_direct = self._dist_to_goal(self._robot_x, self._robot_y)
        if d_direct < 1e-3:
            return None

        d_threshold = d_direct * (1.0 - self._h_margin)
        best: Optional[TangentGap] = None

        cos_y = math.cos(self._robot_yaw)
        sin_y = math.sin(self._robot_yaw)

        for k in range(len(valid_idx) - 1):
            i = valid_idx[k]
            j = valid_idx[k + 1]

            if abs(i - j) > 3:
                continue

            r_near = float(ranges[i])
            r_far  = float(ranges[j])

            # Salto: rayo corto (obstáculo) seguido de rayo largo (espacio libre)
            if r_far < r_near * self._gap_ratio:
                continue

            # [FIX-R2] Ancho real del gap: cuerda angular a distancia r_near
            delta_angle = abs(float(angles[j]) - float(angles[i]))
            # Normalizar por si el salto cruza ±π
            delta_angle = min(delta_angle, 2.0 * math.pi - delta_angle)
            gap_width   = r_near * 2.0 * math.sin(delta_angle / 2.0)

            # [FIX-R1] El gap debe ser transitable por el robot completo
            if gap_width < self._gap_min_w:
                continue

            # Punto tangente: extremo del rayo largo (borde libre del gap)
            tang_angle = float(angles[j])
            tang_x_rob = r_far * math.cos(tang_angle)
            tang_y_rob = r_far * math.sin(tang_angle)

            # Transformar al frame mundo con el yaw actual del robot
            tang_x_w = self._robot_x + cos_y * tang_x_rob - sin_y * tang_y_rob
            tang_y_w = self._robot_y + sin_y * tang_x_rob + cos_y * tang_y_rob

            d_to_tang   = math.hypot(tang_x_rob, tang_y_rob)
            d_tang_goal = math.hypot(self._goal_x - tang_x_w,
                                     self._goal_y - tang_y_w)
            d_heur = d_to_tang + d_tang_goal

            if d_heur >= d_threshold:
                continue

            if best is None or d_heur < best.d_heuristic:
                best = TangentGap(
                    bearing     = tang_angle,
                    gap_x       = tang_x_w,
                    gap_y       = tang_y_w,
                    d_heuristic = d_heur,
                    width_m     = gap_width,
                )

        return best

    # ── Wall-follow con P-controller lateral ──────────────────────────────────

    def _wall_follow_cmd(self, front: float, left: float, right: float) -> Twist:
        """
        [FIX-R3] wall_follow_dist tiene un floor de robot_radius + wall_clearance_m
        aplicado en __init__, por lo que este método no necesita cambiarse.
        El P-controller ya usa self._wf_dist que incluye el floor.
        """
        turn_sign = self._turn_sign
        wall_dist = right if turn_sign > 0 else left

        lat_error = self._wf_dist - wall_dist
        w_lateral = -turn_sign * self._wf_kp * lat_error
        w_lateral = max(-self._wf_w_max, min(self._wf_w_max, w_lateral))

        front_ratio = (front - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
        front_ratio = max(0.1, min(1.0, front_ratio))
        v_linear    = self.ref_v * front_ratio

        # Esquina detectada: parar y girar
        if front < self.emg_d * 1.4:
            v_linear  = 0.0
            w_lateral = turn_sign * self.ref_w

        cmd = Twist()
        cmd.linear.x  = v_linear
        cmd.angular.z = w_lateral
        return cmd

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dist_to_goal(self, x: float, y: float) -> float:
        if self._goal_x is None:
            return float('inf')
        return math.hypot(self._goal_x - x, self._goal_y - y)

    def _maybe_trigger_replan(self, now: float):
        if self._replanning:
            return
        if (now - self._last_replan_ts) < self._replan_cd:
            return
        msg = Pose2D()
        msg.x = self._robot_x
        msg.y = self._robot_y
        self._pub_replan.publish(msg)
        self._replanning     = True
        self._last_replan_ts = now
        self.get_logger().warn(
            f'[TangentBug] Replan desde ({self._robot_x:.2f}, {self._robot_y:.2f})')

    def _sector_min(self, scan: LaserScan, center_rad: float, half_rad: float) -> float:
        """
        Devuelve la distancia mínima al obstáculo más cercano dentro del cono
        definido por center_rad ± half_rad, corregido por lidar_yaw_offset.

        Se usan tres conos:
          - Frente : center=0.0 rad,   half=front_half_deg
          - Izq    : center=+π/2 rad,  half=side_half_deg
          - Der    : center=-π/2 rad,  half=side_half_deg
        """
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = (scan.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
                  + self._lidar_yaw)
        diff = np.arctan2(np.sin(angles - center_rad), np.cos(angles - center_rad))
        mask = (
            (np.abs(diff) <= half_rad) &
            (ranges > scan.range_min) &
            (ranges < scan.range_max)
        )
        valid = ranges[mask]
        return float(np.min(valid)) if valid.size > 0 else float('inf')

    def _bool_msg(self, val: bool) -> Bool:
        m = Bool(); m.data = val; return m

    def _publish(self, cmd: Twist, mode: str):
        if mode != self._mode:
            if mode not in (PASS_THROUGH, PREDICTIVE_BRAKE):
                self.get_logger().warn(
                    f'[TangentBug] {self._mode} → {mode}',
                    throttle_duration_sec=0.4)
            if self._mode == TANGENT_WALL and mode == PASS_THROUGH:
                self._pub_nav_pause.publish(self._bool_msg(False))
            self._mode = mode

        self._pub_cmd.publish(cmd)
        s = String(); s.data = mode
        self._pub_status.publish(s)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TangentBugReflex()
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