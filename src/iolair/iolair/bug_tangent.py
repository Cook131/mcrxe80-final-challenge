#!/usr/bin/env python3
"""
bug_tangent.py  —  Capa Reflejo de Seguridad v5.2  (Tangent Bug + A* Replan + Map Filter)
========================================================================================
Nodo intermedio entre nav_fsm/GoToGoal y puzzlebotController.

Algoritmo:
1. Filtra el /scan contra el /map. Las paredes conocidas se ignoran.
2. Detecta obstáculos DESCONOCIDOS que interrumpen la trayectoria.
3. Al detectar uno, desactiva la navegación global (/nav_pause = True) y asume control.
4. Esquiva el obstáculo mediante heurística Tangent Bug.
5. Al librar el obstáculo, dispara un recálculo de ruta (/replan_trigger).
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry, OccupancyGrid
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
    bearing: float
    gap_x: float
    gap_y: float
    d_heuristic: float
    width_m: float


class TangentBugReflex(Node):
    def __init__(self):
        super().__init__('bug_tangent')

        # ── Parámetros: geometría del robot ───────────────────────────────────
        self.declare_parameter('robot_radius_m',      0.18)
        self.declare_parameter('gap_safety_factor',   1.5)
        self.declare_parameter('wall_clearance_m',    0.08)

        # ── Parámetros heredados ──────────────────────────────────────────────
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
        self.declare_parameter('gap_jump_ratio',      1.30)
        self.declare_parameter('heuristic_margin',    0.10)
        self.declare_parameter('tangent_sector_deg',  120.0)
        self.declare_parameter('min_follow_m',        0.25)

        # ── Leer y validar parámetros ──────────────────────────────────────────
        self._robot_r      = float(self.get_parameter('robot_radius_m').value)
        self._gap_safety   = float(self.get_parameter('gap_safety_factor').value)
        self._wall_clr     = float(self.get_parameter('wall_clearance_m').value)

        self._gap_min_w    = 2.0 * self._robot_r * self._gap_safety

        self.warn_d        = float(self.get_parameter('warn_dist').value)
        self.emg_d         = max(float(self.get_parameter('emergency_dist').value), self._robot_r)
        self.stop_d        = max(float(self.get_parameter('stop_dist').value), 0.05)

        self.ref_v         = float(self.get_parameter('reflex_v').value)
        self.ref_w         = float(self.get_parameter('reflex_w').value)
        self.hold_s        = float(self.get_parameter('reflex_hold_ms').value) / 1000.0
        self.front_h       = math.radians(self.get_parameter('front_half_deg').value)
        self.side_h        = math.radians(self.get_parameter('side_half_deg').value)
        self.hyst          = float(self.get_parameter('hysteresis').value)
        self._lidar_yaw    = float(self.get_parameter('lidar_yaw_offset').value)
        self._replan_cd    = float(self.get_parameter('replan_cooldown_s').value)

        self._wf_dist      = max(float(self.get_parameter('wall_follow_dist').value), self._robot_r + self._wall_clr)
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

        self._hit_x          = 0.0
        self._hit_y          = 0.0
        self._traveled       = 0.0
        self._prev_x         = 0.0
        self._prev_y         = 0.0
        self._turn_sign      = +1.0
        self._best_gap: Optional[TangentGap] = None

        # ── Variables del Mapa Estático ───────────────────────────────────────
        self.grid_map     = None
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_res      = 0.05
        self.map_width    = 0
        self.map_height   = 0

        # ── Suscriptores ──────────────────────────────────────────────────────
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        
        self.create_subscription(OccupancyGrid, '/map',          self._cb_map,   10)
        self.create_subscription(Twist,         '/cmd_raw',      self._cb_cmd,   10)
        self.create_subscription(LaserScan,     '/scan',         self._cb_scan,  scan_qos)
        self.create_subscription(Odometry,      '/odom',         self._cb_odom,  10)
        self.create_subscription(String,        '/astar/status', self._cb_astar, 10)
        self.create_subscription(Pose2D,        '/astar/goal',   self._cb_goal,  10)

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_cmd       = self.create_publisher(Twist,  '/cmd_vel',        10)
        self._pub_status    = self.create_publisher(String, '/reflex_status',  10)
        self._pub_replan    = self.create_publisher(Pose2D, '/replan_trigger', 10)
        self._pub_nav_pause = self.create_publisher(Bool,   '/nav_pause',      10)

        self.create_timer(0.05, self._loop)   # 20 Hz

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        """Procesa el mapa estático para poder filtrar paredes conocidas."""
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_res      = msg.info.resolution
        self.map_width    = msg.info.width
        self.map_height   = msg.info.height
        self.grid_map     = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

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

    def _cb_astar(self, msg: String):
        if msg.data in ('EXECUTING', 'GOAL_REACHED', 'NO_PATH'):
            if self._replanning:
                self.get_logger().info(f'[TangentBug] Replan completado (A*: {msg.data})')
            self._replanning = False

    # ── Lógica de Filtrado ────────────────────────────────────────────────────

    def _filter_scan(self, scan: LaserScan) -> np.ndarray:
        """
        Retorna un array de rangos donde los puntos que coinciden con paredes 
        del mapa estático se vuelven 'inf'. Solo quedan obstáculos desconocidos.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if self.grid_map is None:
            return ranges  # Si no hay mapa, tratar todo como desconocido

        angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment + self._lidar_yaw

        # Proyectar puntos al frame mundo
        x_w = self._robot_x + ranges * np.cos(angles + self._robot_yaw)
        y_w = self._robot_y + ranges * np.sin(angles + self._robot_yaw)

        # Convertir a índices del mapa
        ix = np.floor((x_w - self.map_origin_x) / self.map_res).astype(int)
        iy = np.floor((y_w - self.map_origin_y) / self.map_res).astype(int)

        # Validar fronteras y rangos válidos
        valid_idx = (
            (ix >= 0) & (ix < self.map_width) & 
            (iy >= 0) & (iy < self.map_height) & 
            (ranges > scan.range_min) & (ranges < scan.range_max)
        )

        is_known = np.zeros_like(ranges, dtype=bool)
        
        # Consultar celdas del mapa (Consideramos ocupado si el costo es > 50)
        map_vals = self.grid_map[iy[valid_idx], ix[valid_idx]]
        is_known[valid_idx] = map_vals > 50

        # Crear nuevo array filtrado
        filtered_ranges = ranges.copy()
        filtered_ranges[is_known] = np.inf
        return filtered_ranges

    # ── Loop principal ────────────────────────────────────────────────────────

    def _loop(self):
        scan = self.scan
        if scan is None:
            self._publish(self._last_cmd, PASS_THROUGH)
            return

        # 1. Obtenemos el scan filtrado (solo obstáculos nuevos)
        filtered_ranges = self._filter_scan(scan)

        # 2. Las métricas de distancia usan el scan filtrado
        front = self._sector_min(scan, filtered_ranges, 0.0,               self.front_h)
        left  = self._sector_min(scan, filtered_ranges, math.radians( 90), self.side_h)
        right = self._sector_min(scan, filtered_ranges, math.radians(-90), self.side_h)

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

        # ── P2: TANGENT_WALL (Evasión) ────────────────────────────────────────
        in_wall = (self._mode == TANGENT_WALL)
        emg_thr = self.emg_d + (self.hyst if in_wall else 0.0)

        if front <= emg_thr or (hold_active and in_wall):
            if not in_wall:
                self._enter_wall_follow(left, right, now)
            else:
                step = math.hypot(self._robot_x - self._prev_x, self._robot_y - self._prev_y)
                self._traveled += step
                self._prev_x    = self._robot_x
                self._prev_y    = self._robot_y

                if self._traveled >= self._min_follow:
                    gap = self._best_tangent_gap(scan, filtered_ranges)
                    if gap is not None:
                        self.get_logger().info(f'[TangentBug] Evasión terminada, saliendo por gap.')
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
            self.get_logger().info(f'[TangentBug] {self._mode} → PASS (Despejado)')
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

        # 3. Desactivar la navegación global y tomar el control
        self._pub_nav_pause.publish(self._bool_msg(True))
        self.get_logger().warn(f'[TangentBug] Obstáculo DESCONOCIDO detectado. Nav Pausada. Iniciando evasión.')

    # ── Heurística Tangent Bug ────────────────────────────────────────────────

    def _best_tangent_gap(self, scan: LaserScan, filtered_ranges: np.ndarray) -> Optional[TangentGap]:
        if self._goal_x is None:
            return None

        ranges = filtered_ranges # Usamos el scan ya sin el mapa estático
        angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment + self._lidar_yaw
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

            # Si r_near es inf, no hay obstáculo
            if math.isinf(r_near):
                continue

            # Detectar discontinuidad (gap)
            if r_far < r_near * self._gap_ratio:
                continue

            delta_angle = abs(float(angles[j]) - float(angles[i]))
            delta_angle = min(delta_angle, 2.0 * math.pi - delta_angle)
            gap_width   = r_near * 2.0 * math.sin(delta_angle / 2.0)

            if gap_width < self._gap_min_w:
                continue

            tang_angle = float(angles[j])
            # Si el rayo libre r_far es inf, usar un rango nominal para proyectar
            proj_r = r_far if not math.isinf(r_far) else scan.range_max
            tang_x_rob = proj_r * math.cos(tang_angle)
            tang_y_rob = proj_r * math.sin(tang_angle)

            tang_x_w = self._robot_x + cos_y * tang_x_rob - sin_y * tang_y_rob
            tang_y_w = self._robot_y + sin_y * tang_x_rob + cos_y * tang_y_rob

            d_to_tang   = math.hypot(tang_x_rob, tang_y_rob)
            d_tang_goal = math.hypot(self._goal_x - tang_x_w, self._goal_y - tang_y_w)
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

    # ── Control y Helpers ─────────────────────────────────────────────────────

    def _wall_follow_cmd(self, front: float, left: float, right: float) -> Twist:
        turn_sign = self._turn_sign
        wall_dist = right if turn_sign > 0 else left

        # Si perdemos la pared temporalmente (ej. por ser inf), limitamos el error
        if math.isinf(wall_dist): wall_dist = self._wf_dist * 1.5

        lat_error = self._wf_dist - wall_dist
        w_lateral = -turn_sign * self._wf_kp * lat_error
        w_lateral = max(-self._wf_w_max, min(self._wf_w_max, w_lateral))

        front_val = front if not math.isinf(front) else self.warn_d
        front_ratio = (front_val - self.emg_d) / max(self.warn_d - self.emg_d, 1e-6)
        front_ratio = max(0.1, min(1.0, front_ratio))
        v_linear    = self.ref_v * front_ratio

        if front < self.emg_d * 1.4:
            v_linear  = 0.0
            w_lateral = turn_sign * self.ref_w

        cmd = Twist()
        cmd.linear.x  = v_linear
        cmd.angular.z = w_lateral
        return cmd

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
        self.get_logger().warn(f'[TangentBug] Disparando REPLANIFICACIÓN desde ({self._robot_x:.2f}, {self._robot_y:.2f})')

    def _sector_min(self, scan: LaserScan, filtered_ranges: np.ndarray, center_rad: float, half_rad: float) -> float:
        angles = scan.angle_min + np.arange(len(filtered_ranges), dtype=np.float32) * scan.angle_increment + self._lidar_yaw
        diff = np.arctan2(np.sin(angles - center_rad), np.cos(angles - center_rad))
        mask = (
            (np.abs(diff) <= half_rad) &
            (filtered_ranges > scan.range_min) &
            (filtered_ranges < scan.range_max)
        )
        valid = filtered_ranges[mask]
        return float(np.min(valid)) if valid.size > 0 else float('inf')

    def _bool_msg(self, val: bool) -> Bool:
        m = Bool(); m.data = val; return m

    def _publish(self, cmd: Twist, mode: str):
        if mode != self._mode:
            if mode not in (PASS_THROUGH, PREDICTIVE_BRAKE):
                self.get_logger().warn(f'[TangentBug] {self._mode} → {mode}', throttle_duration_sec=0.4)
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